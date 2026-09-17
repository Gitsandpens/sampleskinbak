"""
CNN model for skin disease classification.

Classes follow the HAM10000 dermatology dataset convention (7 lesion types).
Swap in your own dataset/classes as needed -- just update CLASS_NAMES and
the final Linear layer's out_features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = [
    "akiec",  # Actinic keratoses / intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
]

CLASS_INFO = {
    "akiec": {"full_name": "Actinic Keratoses", "risk": "high", "advice": "Precancerous — see a dermatologist promptly."},
    "bcc":   {"full_name": "Basal Cell Carcinoma", "risk": "high", "advice": "Most common skin cancer — needs medical evaluation."},
    "bkl":   {"full_name": "Benign Keratosis-like Lesion", "risk": "low", "advice": "Usually harmless, but monitor for changes."},
    "df":    {"full_name": "Dermatofibroma", "risk": "low", "advice": "Benign, typically no treatment needed."},
    "mel":   {"full_name": "Melanoma", "risk": "critical", "advice": "Potentially dangerous — seek immediate medical attention."},
    "nv":    {"full_name": "Melanocytic Nevi", "risk": "low", "advice": "Common mole, low risk. Keep an eye on changes (ABCDE rule)."},
    "vasc":  {"full_name": "Vascular Lesion", "risk": "low", "advice": "Usually benign blood-vessel related lesion."},
}


class SkinDiseaseCNN(nn.Module):
    """A compact CNN suitable for federated training (small, fast to
    train locally on each simulated client's shard)."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
