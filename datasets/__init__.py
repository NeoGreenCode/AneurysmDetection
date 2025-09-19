
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
from typing import List, Tuple


TARGET_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery', 
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

def get_windowing_params(modality: str) -> Tuple[float, float]:
    windows = {'CT': (40, 80),'CTA': (50, 350), 'MRA': (600, 1200), 'MRI': (40, 80), 'MR': (40, 80)}
    return windows.get(modality, (40, 80))

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)


def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
    if not config.USE_CLAHE:
        return img
    if modality in ['CTA', 'MRA']:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    elif modality in ['MRI', 'MR']:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = np.power(img_clahe / 255.0, 0.9) * 255
        img_clahe = img_clahe.astype(np.uint8)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
    return img_clahe

def robust_normalization(volume: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(volume.flatten(), [1, 99])
    volume_norm = np.clip(volume, p1, p99)
    if p99 > p1:
        volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)
    else:
        volume_norm = np.zeros_like(volume_norm)
    return (volume_norm * 255).astype(np.uint8)

class FrameDataset(Dataset):
    """Dataset usando apenas DICOMs"""
    def __init__(self, df, num_frames=8, transform=None, 
                 CACHE_SIZE=100, USE_METADATA=True, DICOM_DIR:str=None, IMAGE_SIZE:int=224,
                 USE_WINDOWING:bool=True):
        
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.transform = transform
        self._cache = {}
        self._cache_keys = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._max_cache_size = CACHE_SIZE
        self.USE_METADATA = USE_METADATA
        self.DICOM_DIR = DICOM_DIR
        self.IMAGE_SIZE = IMAGE_SIZE
        self.USE_WINDOWING  = USE_WINDOWING

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']

        labels = torch.tensor(row[TARGET_COLS].values.astype(np.float32))  # CPU
        metadata = self._extract_metadata(row)  # CPU
        image = self._load_volume_dicom(series_uid, row)  # CPU

        result = (image, labels, metadata)
        self._update_cache(idx, result)
        return result
        

    def _update_cache(self, idx, data):
        if len(self._cache) >= self._max_cache_size:
            oldest_idx = self._cache_keys.pop(0)
            del self._cache[oldest_idx]
        self._cache[idx] = data
        self._cache_keys.append(idx)
        
    def _extract_metadata(self, row):
        if not self.USE_METADATA:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)
        age = row.get('PatientAge', 50)
        if pd.isna(age):
            age = 50
        elif isinstance(age, str):
            age = int(''.join(filter(str.isdigit, age[:3])) or '50')
        age = min(float(age), 100.0) / 100.0
        sex = row.get('PatientSex', 'M')
        sex = 1.0 if sex == 'M' else 0.0
        return torch.tensor([age, sex], dtype=torch.float32)
    
    def _load_volume_dicom(self, series_uid: str, row):
        dicom_dir = os.path.join(self.DICOM_DIR, series_uid)
        if not os.path.exists(dicom_dir):
            return torch.zeros((3, self.num_frames, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=torch.float32)

        dicom_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
        if len(dicom_files) == 0:
            return torch.zeros((3, self.num_frames, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=torch.float32)

        total_slices = len(dicom_files)
        indices = np.linspace(0, total_slices - 1, self.num_frames).astype(int)
        sampled_files = [dicom_files[i] for i in indices]

        volume = []
        modality = row.get('Modality', 'CT')
        for file in sampled_files:
            try:
                ds = pydicom.dcmread(file)
                img = ds.pixel_array.astype(np.float32)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * ds.RescaleSlope + ds.RescaleIntercept
                if self.USE_WINDOWING:
                    wc, ww = get_windowing_params(modality)
                    img = apply_dicom_windowing(img, wc, ww)
                img = apply_clahe_normalization(img, modality)

                # code backup:
                img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                volume.append(img)
                
            except Exception:
                volume.append(np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8))

        volume = np.stack(volume, axis=0)
        volume = robust_normalization(volume)

        mip = np.max(volume, axis=0)
        std_proj = np.std(volume, axis=0)
        frames = []
        for i in range(volume.shape[0]):
            middle_slice = volume[i]
            three_channel = np.stack([middle_slice, mip, std_proj], axis=-1).astype(np.uint8)
            if self.transform:
                three_channel = self.transform(image=three_channel)["image"]
            frames.append(three_channel)

        frames = torch.stack(frames, dim=0).permute(1,0,2,3)
        return frames
    
