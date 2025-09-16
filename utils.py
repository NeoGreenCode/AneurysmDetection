import numpy as np
import cv2
import albumentations as A


def _channel1(tensor: np.ndarray) -> np.ndarray:
    """
    Converts a 4D tensor (C, D, H, W) to grayscale if needed.

    Args:
        tensor (np.ndarray): Input tensor with shape (C, D, H, W).

    Returns:
        np.ndarray: Grayscale tensor with shape (1, D, H, W) and dtype float32.
    """
    tensor = np.array(tensor, dtype=np.float32)
    # If already grayscale (C=1), return as is
    if tensor.shape[0] == 1:
        return tensor
    # If C > 1, average across channels for each frame (D)
    # Result: (1, D, H, W)
    gray = np.mean(tensor, axis=0, keepdims=False)  # (D, H, W)
    gray = np.expand_dims(gray, axis=0)             # (1, D, H, W)
    return gray.astype(np.float32)


def _channel2(volume: np.ndarray) -> np.ndarray:
    """
    Generates an edge-enhanced channel using Gaussian blur for each slice.

    Args:
        volume (np.ndarray): Input volume with shape (D, H, W).

    Returns:
        np.ndarray: Edge-enhanced volume with shape (D, H, W) and dtype float32.
    """
    channel2 = []
    for slice_img in volume:
        # Apply Gaussian blur to each slice
        img_blur = cv2.GaussianBlur(slice_img.astype(np.float32), (3,3), 0)
        channel2.append(img_blur)
    return np.array(channel2, dtype=np.float32)


def _channel3(volume: np.ndarray) -> np.ndarray:
    """
    Generates an attention map channel using gradient magnitude for each slice.

    Args:
        volume (np.ndarray): Input volume with shape (D, H, W).

    Returns:
        np.ndarray: Attention map volume with shape (D, H, W) and dtype float32.
    """
    att_vol = []
    for img in volume:
        img = img.astype(np.float32)
        # Compute gradients in x and y directions
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        # Compute gradient magnitude
        att_map = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize to 0-255
        att_map = 255 * (att_map / (np.max(att_map) + 1e-6))
        att_vol.append(att_map.astype(np.float32))
    return np.array(att_vol, dtype=np.float32)


def apply_transform(vol_stacked: np.ndarray) -> np.ndarray:
    """
    Applies a set of augmentation transformations to each slice in the volume.

    Args:
        vol_stacked (np.ndarray): Input volume with shape (D, H, W).

    Returns:
        np.ndarray: Augmented volume with shape (D, H, W) and dtype float32.
    """
    # Define augmentation pipeline
    transform = A.Compose([
        A.Rotate(limit=15, p=0.7),  # Random rotation
        A.HorizontalFlip(p=0.5),    # Random horizontal flip
        A.Affine(translate_percent=0.05, scale=(0.9,1.1), rotate=(-10,10), p=0.5),
        A.ElasticTransform(alpha=30, sigma=4, p=0.3),
        A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
        A.CoarseDropout(p=0.2),
    ])
    transformed_volume = []
    for slice_img in vol_stacked:
        slice_img = slice_img.astype(np.float32)
        # Apply augmentations to each slice
        augmented = transform(image=slice_img)['image']
        transformed_volume.append(augmented)
    return np.array(transformed_volume, dtype=np.float32)