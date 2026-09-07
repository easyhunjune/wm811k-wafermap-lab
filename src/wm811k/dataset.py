from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import LABEL_TO_INDEX
from .preprocessing import encode_wafer_map


class WaferDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        mode: str,
        output_size: int,
    ) -> None:
        required = {"waferMap", "failure_label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame is missing columns: {sorted(missing)}")
        self.wafer_maps = frame["waferMap"].tolist()
        self.targets = frame["failure_label"].map(LABEL_TO_INDEX).to_numpy()
        self.mode = mode
        self.output_size = output_size

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = encode_wafer_map(
            self.wafer_maps[index],
            output_size=self.output_size,
            mode=self.mode,
            validate_values=False,
        )
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return inputs, target
