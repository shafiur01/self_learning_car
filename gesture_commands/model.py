"""Small temporal neural network: learns patterns across frames."""

import torch
from torch import nn

from .data import CONVENTION, FEATURES, FRAMES, LABELS, SECONDS, VERSION


class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(FEATURES, 48, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2), nn.ReLU(),
            # Preserve coarse time order; inward and outward motion must differ.
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(64 * 4, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, len(LABELS)),
        )

    def forward(self, x):
        return self.network(x.transpose(1, 2))


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema_version": VERSION, "convention": CONVENTION,
                "labels": list(LABELS), "frames": FRAMES, "seconds": SECONDS}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Model and application settings differ. Retrain with this version.")
    model = GestureNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
