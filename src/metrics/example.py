import torch
import numpy as np
from sklearn.metrics import roc_curve
from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preds = []
        self.targets = []
    
    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        self.preds.append(preds.detach().cpu()[0])
        self.targets.append(targets.detach().cpu())

    def __call__(self, **kwargs):
        preds = torch.cat(self.preds).numpy()
        targets = torch.cat(self.targets).numpy()

        fpr, tpr, threshold = roc_curve(targets, preds, pos_label=0)
        fnr = 1 - tpr
        _ = threshold[np.argmin(abs(fnr - fpr))]

        EER_fpr = fpr[np.argmin(np.absolute((fnr - fpr)))]
        EER_fnr = fnr[np.argmin(np.absolute((fnr - fpr)))]
        EER = 0.5 * (EER_fpr + EER_fnr)

        return EER
