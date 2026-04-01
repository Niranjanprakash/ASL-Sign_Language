"""
model.py - MLP model definition for ASL alphabet classification.
Input: concatenated [normalized_landmarks(63) + geometric_features(N_geo)]
Output: NUM_CLASSES logits
"""

import torch
import torch.nn as nn
from utils import NUM_CLASSES


def get_input_size(sample_feature_size: int = 63) -> int:
    return 63


class ASLMLP(nn.Module):
    """
    Lightweight MLP for ASL alphabet recognition.
    Architecture exactly matches requirements:
    Input(63) → 128 → 64 → NUM_CLASSES
    """

    def __init__(self, input_size: int = 63, num_classes: int = NUM_CLASSES):
        super(ASLMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(path: str, input_size: int = 63, num_classes: int = NUM_CLASSES,
               device: str = "cpu") -> ASLMLP:
    """Load a saved model checkpoint."""
    model = ASLMLP(input_size=input_size, num_classes=num_classes)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model
