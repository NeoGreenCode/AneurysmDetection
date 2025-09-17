# Running locally
try:
    from architecture import arch_r3d18 
    from utils import _channel1, _channel2, _channel3

# Running as module
except ImportError:
    from .architecture import arch_r3d18
    from .utils import _channel1, _channel2, _channel3

# Import necessary libraries
import torch
import numpy as np
import pandas as pd
import gc
from huggingface_hub import hf_hub_download


# Dictionary of available models and their configurations
global available_models, LABEL_COLS
available_models = {
    "R3D18_AneurysmDetection_33M": {
        "num_classes": 14,
        "num_frames": 16,
        "model": arch_r3d18
    }
}

LABEL_COLS = [
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

class AneurysmDetection:
    """
    Main class for aneurysm detection using a 3D ResNet-based model.
    Handles model loading, inference, and preprocessing.
    """

    def __init__(self, model_name: str = "R3D18_AneurysmDetection_33M"):
        """
        Initializes the AneurysmDetection class.

        Args:
            model_name (str): Name of the model to use. Must be present in available_models.
        """
        
        self.model_name = model_name    
        if self.model_name not in available_models:
            raise ValueError(f"Model '{self.model_name}' not found in {available_models.keys()}.")


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download or set the model path before building the model
        self.model_download()
        self.build_model()

    def model_download(self):
        """
        Downloads the model weights from HuggingFace Hub.

        Raises:
            ValueError: If the model_name is not in available_models.
            Exception: For any error during download.
        """
        

        try:
            print(f"Downloading model from Hugging Face Hub...")

            self.model_path = hf_hub_download(
                repo_id="claytonsds/R3D18_AneurysmDetection_33M",
                filename=f"{self.model_name}.pth"
            )

            print("Model downloaded successfully.")

        except Exception as e:
            print(f"Error downloading model: {e}")
            raise e

    def build_model(self):
        """
        Builds the model architecture and loads the pretrained weights.
        """
        self.model = available_models[self.model_name]["model"](
            num_frames=available_models[self.model_name]["num_frames"], 
            num_classes=available_models[self.model_name]["num_classes"],
            pretrained=False
        ).to(self.device)

        # Load checkpoint and extract model_state
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            
        else:
            raise KeyError("Could not find 'model_state' in checkpoint.")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Model loaded and ready for inference.")

    def predict_one_batch(self, input_tensor: np.ndarray) -> pd.DataFrame:
        """
        Runs inference on a single 4D input tensor.

        Args:
            input_tensor (np.ndarray or torch.Tensor): Input tensor with shape (C, D, H, W).

        Returns:
            pd.DataFrame: Model output probabilities for the input.
        """
        # Accept both numpy arrays and torch tensors
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.cpu().numpy()

        channel1 = _channel1(input_tensor)[0]  # (D, H, W)
        channel2 = _channel2(input_tensor[0])  # (D, H, W)
        channel3 = _channel3(input_tensor[0])  # (D, H, W)

        vol_stacked = np.stack([channel1, channel2, channel3], axis=0)  # (3, D, H, W)
        vol_tensor = torch.from_numpy(vol_stacked).float().unsqueeze(0).to(self.device)  # (1, 3, D, H, W)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(vol_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return pd.DataFrame([probs.tolist()], columns=LABEL_COLS)

    def predict_batch(self, input_tensors: list) -> pd.DataFrame:
        """
        Runs inference on a batch of input tensors.

        Args:
            input_tensors (list): List of input tensors, each with shape (C, D, H, W) as np.ndarray or torch.Tensor.

        Returns:
            pd.DataFrame: Model output probabilities for the batch, each row corresponds to an input.
        """
        batch_tensors = []
        for tensor in input_tensors:
            # Accept both numpy arrays and torch tensors
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()
            channel1 = _channel1(tensor)[0]  # (D, H, W)
            channel2 = _channel2(tensor[0])  # (D, H, W)
            channel3 = _channel3(tensor[0])  # (D, H, W)
            vol_stacked = np.stack([channel1, channel2, channel3], axis=0)  # (3, D, H, W)
            batch_tensors.append(vol_stacked)
        
        batch_array = np.stack(batch_tensors, axis=0)  # (B, 3, D, H, W)
        batch_tensor = torch.from_numpy(batch_array).float().to(self.device)  # (B, 3, D, H, W)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, num_classes)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return pd.DataFrame(probs.tolist(), columns=LABEL_COLS)

    def predict_tensor(self, input_tensor) -> pd.DataFrame:
        """
        Runs inference on a single or batch of 4D input tensors.

        Args:
            input_tensor (np.ndarray, torch.Tensor, or list): 
                - Single tensor with shape (C, D, H, W)
                - Batch tensor with shape (B, C, D, H, W)
                - List of tensors, each with shape (C, D, H, W)

        Returns:
            pd.DataFrame: Model output probabilities for each input (one row per input).
        """
        # Handle torch.Tensor input
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.cpu().numpy()

        # If input is a single tensor (C, D, H, W), wrap in a list
        if isinstance(input_tensor, np.ndarray) and input_tensor.ndim == 4:
            input_tensor = [input_tensor]
        # If input is a batch tensor (B, C, D, H, W), convert to list of tensors
        elif isinstance(input_tensor, np.ndarray) and input_tensor.ndim == 5:
            input_tensor = [input_tensor[i] for i in range(input_tensor.shape[0])]
        # If input is already a list, do nothing

        batch_tensors = []
        for tensor in input_tensor:
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()
            channel1 = _channel1(tensor)[0]  # (D, H, W)
            channel2 = _channel2(tensor[0])  # (D, H, W)
            channel3 = _channel3(tensor[0])  # (D, H, W)
            vol_stacked = np.stack([channel1, channel2, channel3], axis=0)  # (3, D, H, W)
            batch_tensors.append(vol_stacked)

        batch_array = np.stack(batch_tensors, axis=0)  # (B, 3, D, H, W)
        batch_tensor = torch.from_numpy(batch_array).float().to(self.device)  # (B, 3, D, H, W)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, num_classes)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return pd.DataFrame(probs.tolist(), columns=LABEL_COLS)
