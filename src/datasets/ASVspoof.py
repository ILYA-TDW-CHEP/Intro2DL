import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVspoofDataSet(BaseDataset):
    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "ASVspoofData" / "LA" / "LA" / f"ASVspoof2019_LA_{name}" / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.  
        """
        index = []
        data_path = ROOT_PATH / "data" / "ASVspoofData" / "LA" / "LA"
    
        data_path.mkdir(exist_ok=True, parents=True)

        protocol_path = data_path / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{name}.trl.txt"
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                filename = f"{parts[1]}.flac"
                label = 1 if parts[-1] == "bonafide" else 0 
                index.append({
                    "path": str(data_path) + f"/ASVspoof2019_LA_{name}/flac/" + filename,
                    "label": label
                })

        write_json(index, str(data_path / f"index_{name}.json"))

        return index
