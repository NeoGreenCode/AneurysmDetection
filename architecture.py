import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class arch_r3d18(nn.Module):
    """
    3D ResNet-based model with a custom classifier for multi-frame volumetric inputs.

    Args:
        num_frames (int): Number of frames (depth) expected in the input.
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to use pretrained weights for the backbone.
    """
    
    def __init__(self, num_frames: int = 16, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # Use pretrained weights if specified
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        self.backbone.fc = nn.Identity()  # Remove original classification layer
        
        self.feature_dim = 512  # Output feature dimension of r3d_18
        print(f"Backbone r3d_18 (Conv3D): {self.feature_dim} features")
        
        # Custom classifier head for the model
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W], where
                B = batch size,
                C = number of channels,
                D = number of frames (depth),
                H = height,
                W = width.

        Returns:
            torch.Tensor: Output logits of shape [B, num_classes].
        """
        features = self.backbone(x)
        return self.classifier(features)