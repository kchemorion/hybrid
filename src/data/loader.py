import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import scipy.special
from sklearn.preprocessing import StandardScaler
import h5py

class MissingValueDataset(Dataset):
    """Dataset for handling data with missing values."""

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        mask: Optional[np.ndarray] = None,
        categorical_cols: Optional[List[int]] = None,
        augmentation: bool = False,
        missing_mechanism: str = "MCAR",
        missing_ratio: float = 0.2,
    ):
        self.missing_mechanism = missing_mechanism
        self.missing_ratio = missing_ratio
        self.augmentation = augmentation
        self.categorical_cols = categorical_cols

        if isinstance(data, pd.DataFrame):
            self.data = data.values
            if mask is None:
                self.mask = self._generate_missing_mask(self.data)
            else:
                self.mask = mask.values
            if self.categorical_cols is None:
                self.categorical_cols = self._identify_categorical_columns(data)
        else:
            self.data = data
            if mask is None:
                self.mask = self._generate_missing_mask(self.data)
            else:
                self.mask = mask
            if self.categorical_cols is None:
                self.categorical_cols = []


        self.scalers = self._initialize_scalers()
        self.scaled_data = self._scale_data()

    def _identify_categorical_columns(self, data: pd.DataFrame) -> List[int]:
        """Identify categorical columns."""
        categorical_cols = []
        for i, col in enumerate(data.columns):
            if data[col].dtype == object or data[col].dtype == "category" or len(data[col].unique()) < 10:
                categorical_cols.append(i)
        return categorical_cols

    def _initialize_scalers(self) -> Dict[int, StandardScaler]:
        """Initialize scalers for numerical columns."""
        scalers = {}
        for i in range(self.data.shape[1]):
            if i not in self.categorical_cols:
                scaler = StandardScaler()
                scaler.fit(self.data[:, i].reshape(-1, 1))
                scalers[i] = scaler
        return scalers

    def _scale_data(self) -> np.ndarray:
        """Scale numerical features."""
        scaled_data = self.data.copy()
        for i, scaler in self.scalers.items():
            scaled_data[:, i] = scaler.transform(scaled_data[:, i].reshape(-1, 1)).ravel()
        return scaled_data

    def _generate_missing_mask(self, data: np.ndarray) -> np.ndarray:
        """Generate missing value mask based on the specified mechanism."""
        mask = np.ones_like(data, dtype=bool)
        if self.missing_mechanism == "MCAR":
            mask = np.random.rand(*data.shape) > self.missing_ratio
        elif self.missing_mechanism == "MAR":
            #Implementation for MAR is complex and depends on the data
            raise NotImplementedError("MAR missing mechanism not fully implemented.")
        elif self.missing_mechanism == "MNAR":
            #Implementation for MNAR is complex and depends on the data
            raise NotImplementedError("MNAR missing mechanism not fully implemented.")
        return mask.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.FloatTensor(self.scaled_data[idx])
        mask = torch.FloatTensor(self.mask[idx])
        if self.augmentation:
            data, mask = self._augment_data(data, mask)
        return data, mask

    def _augment_data(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation."""
        for i in range(len(data)):
            if i not in self.categorical_cols and mask[i] == 1:
                data[i] += torch.randn(1).item() * 0.1
        return data, mask

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """Create DataLoader."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

#The StreamingDataset is removed for brevity as it adds complexity without core functionality changes.  It's easily re-added if needed.