import torch
from torch import nn
from torch.nn import Sequential


class MFM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if x.shape[1] % 2 != 0 and x.ndim == 3:
            x = x[:, :-1, :]
            x = x.unsqueeze(0)
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.max(x1, x2)


class LCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ConvPart = Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding='same'),   # Layer 1
            MFM(),                                                                                # Layer 2
            nn.MaxPool2d(kernel_size=2, stride=2),                                                # Layer 3
            nn.Dropout2d(p=0.15),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),                  # Layer 4
            MFM(),                                                                                # Layer 5
            nn.BatchNorm2d(32),                                                                   # Layer 6
            nn.Dropout2d(p=0.15),

            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=3, stride=1, padding='same'),  # Layer 7
            MFM(),                                                                                # Layer 8
            nn.MaxPool2d(kernel_size=2, stride=2),                                                # Layer 9
            nn.BatchNorm2d(48),                                                                   # Layer 10

            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=1),                  # Layer 11
            MFM(),                                                                                # Layer 12
            nn.BatchNorm2d(48),                                                                   # Layer 13

            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding='same'),  # Layer 14
            MFM(),                                                                                # Layer 15
            nn.MaxPool2d(kernel_size=2, stride=2),                                                # Layer 16
            nn.Dropout2d(p=0.15),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),                 # Layer 17
            MFM(),                                                                                # Layer 18
            nn.BatchNorm2d(64),                                                                   # Layer 19

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),  # Layer 20
            MFM(),                                                                                # Layer 21
            nn.BatchNorm2d(32),                                                                   # Layer 22

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),                  # Layer 23
            MFM(),                                                                                # Layer 24
            nn.BatchNorm2d(32),                                                                   # Layer 25
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),  # Layer 26
            MFM(),                                                                                # Layer 27
            nn.MaxPool2d(kernel_size=2, stride=2),                                                # Layer 28
        )

        self.LinearPart = Sequential(
            nn.Linear(in_features=32 * 54 * 37, out_features=160),                              # Layer 29
            nn.Dropout(p=0.15),
            MFM(),                                                                                # Layer 30
            nn.BatchNorm1d(80),                                                                   # Layer 31
            nn.Linear(in_features=80, out_features=2)                                             # Layer 32
        )

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        data_object = self.ConvPart(data_object)
        data_object = data_object.view(data_object.size(0), -1)
        data_object = self.LinearPart(data_object)
        return {"logits": data_object}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
