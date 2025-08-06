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

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        fpr, tpr, threshold = roc_curve(labels, scores, pos_label=0)
        fnr = 1 - tpr
        _ = threshold[np.argmin(abs(fnr - fpr))]

        EER_fpr = fpr[np.argmin(np.absolute((fnr - fpr)))]
        EER_fnr = fnr[np.argmin(np.absolute((fnr - fpr)))]
        EER = 0.5 * (EER_fpr + EER_fnr)

        return EER
