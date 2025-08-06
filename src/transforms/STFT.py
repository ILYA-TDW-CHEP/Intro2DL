import torchaudio
import torch
from torch import nn


class STFT_transform(nn.Module):
    """
    Use STFT transform and some padding things
    """

    def __init__(self):
        super().__init__()
        STFT_CONFIG = {
            "n_fft": 1732,
            "win_length": 512,
            "hop_length": 256,
            "power": None,
            "normalized": False
        }
        self.STFT = torchaudio.transforms.Spectrogram(**STFT_CONFIG)

    def forward(self, x):
        x = self.STFT(x).abs()
        magnitude = x.mean(dim=0, keepdim=True)

        goal_T = 600
        current_T = magnitude.shape[-1]
        pad_T = max(0, goal_T - current_T)

        magnitude = torch.nn.functional.pad(magnitude, (0, pad_T, 0, 0))
        magnitude = magnitude[:, :, :goal_T]
        magnitude = magnitude.unsqueeze(1)
        
        return magnitude
