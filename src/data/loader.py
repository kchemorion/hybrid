import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import scipy.special
from sklearn.preprocessing import StandardScaler
import h5py

class MissingValueDataset(Dataset):
    """
    Dataset class for handling data with missing values.
    
    Features:
    - Supports multiple missing mechanisms (MCAR, MAR, MNAR)
    - Handles mixed data types
    - Provides data augmentation
    - Supports streaming for large datasets
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        missing_mechanism: str = 'MCAR',
        missing_ratio: float = 0.2,
        categorical_columns: Optional[List[int]] = None,
        augmentation: bool = False
    ):
        self.missing_mechanism = missing_mechanism
        self.missing_ratio = missing_ratio
        self.categorical_columns = categorical_columns or []
        self.augmentation = augmentation
        
        # Convert data to numpy array if necessary
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = data
            
        # Initialize scalers
        self.scalers = self._initialize_scalers()
        
        # Create missing value mask
        self.mask = self._generate_missing_mask()
        
        # Scale data
        self.scaled_data = self._scale_data()

    def _identify_categorical_columns(self, data: pd.DataFrame) -> List[int]:
        """Identify categorical columns from DataFrame."""
        if not isinstance(data, pd.DataFrame):
            return self.categorical_columns or []
            
        categorical_cols = []
        for i, col in enumerate(data.columns):
            if data[col].dtype == 'object' or data[col].dtype == 'category' or \
            len(data[col].unique()) < self.categorical_threshold:
                categorical_cols.append(i)
        return categorical_cols
        
    def _initialize_scalers(self) -> Dict[int, StandardScaler]:
        """Initialize scalers for numerical columns."""
        scalers = {}
        for col in range(self.data.shape[1]):
            if col not in self.categorical_columns:
                scaler = StandardScaler()
                scaler.fit(self.data[:, col].reshape(-1, 1))
                scalers[col] = scaler
        return scalers
    
    def _scale_data(self) -> np.ndarray:
        """Scale numerical features while preserving categorical ones."""
        scaled_data = self.data.copy()
        for col, scaler in self.scalers.items():
            scaled_data[:, col] = scaler.transform(
                scaled_data[:, col].reshape(-1, 1)
            ).ravel()
        return scaled_data
    
    def _generate_missing_mask(self) -> np.ndarray:
        """Generate missing value mask based on specified mechanism."""
        mask = np.ones_like(self.data)
        
        if self.missing_mechanism == 'MCAR':
            # Missing Completely at Random
            mask = np.random.rand(*self.data.shape) > self.missing_ratio
            
        elif self.missing_mechanism == 'MAR':
            # Missing at Random
            for col in range(self.data.shape[1]):
                # Make missingness depend on other columns
                predictor_col = (col + 1) % self.data.shape[1]
                predictor_values = self.data[:, predictor_col]
                missing_prob = scipy.special.expit(
                    (predictor_values - predictor_values.mean()) / predictor_values.std()
                )
                mask[:, col] = np.random.rand(len(self.data)) > (missing_prob * self.missing_ratio)
                
        elif self.missing_mechanism == 'MNAR':
            # Missing Not at Random
            for col in range(self.data.shape[1]):
                values = self.data[:, col]
                # Higher values have higher probability of being missing
                missing_prob = scipy.special.expit(
                    (values - values.mean()) / values.std()
                )
                mask[:, col] = np.random.rand(len(self.data)) > (missing_prob * self.missing_ratio)
                
        return mask.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.FloatTensor(self.scaled_data[idx])
        mask = torch.FloatTensor(self.mask[idx])
        
        if self.augmentation:
            data, mask = self._augment_data(data, mask)
            
        return data, mask
    
    def _augment_data(
        self,
        data: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation techniques."""
        # Add random noise to numerical features
        for col in range(len(data)):
            if col not in self.categorical_columns and mask[col] == 1:
                data[col] += torch.randn(1).item() * 0.1
                
        return data, mask
    
    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Create DataLoader with specified parameters."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

class StreamingDataset(MissingValueDataset):
    """Dataset class for handling large datasets that don't fit in memory."""
    
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        missing_mechanism: str = 'MCAR',
        missing_ratio: float = 0.2,
        categorical_columns: Optional[List[int]] = None,
        categorical_threshold: int = 10,  # Add this parameter
        augmentation: bool = False
    ):
        self.missing_mechanism = missing_mechanism
        self.missing_ratio = missing_ratio
        self.categorical_threshold = categorical_threshold
        self.augmentation = augmentation
        
        # Identify categorical columns if not provided
        if isinstance(data, pd.DataFrame):
            self.feature_names = list(data.columns)
            self.categorical_columns = categorical_columns or self._identify_categorical_columns(data)
            self.data = data.values
        else:
            self.feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
            self.categorical_columns = categorical_columns or []
            self.data = data
        
        # Initialize scalers
        self.scalers = self._initialize_scalers()
        
        # Create missing value mask
        self.mask = self._generate_missing_mask()
        
        # Scale data
        self.scaled_data = self._scale_data()
        
    def _load_chunk(self, start: int, end: int) -> np.ndarray:
        """Load a chunk of data from disk."""
        return self.file['data'][start:end]
    
    def __len__(self) -> int:
        return self.data_shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate chunk index
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        # Load appropriate chunk
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.data_shape[0])
        chunk_data = self._load_chunk(start, end)
        
        # Get item from chunk
        data = torch.FloatTensor(self.scaled_data[local_idx])
        mask = torch.FloatTensor(self.mask[local_idx])
        
        if self.augmentation:
            data, mask = self._augment_data(data, mask)
            
        return data, mask
    
    def __del__(self):
        """Close file handle on deletion."""
        self.file.close()

def get_feature_info(self) -> Dict:
    """Get information about features."""
    return {
        'feature_names': self.feature_names,
        'categorical_columns': self.categorical_columns,
        'n_features': self.data.shape[1],
        'n_categorical': len(self.categorical_columns),
        'n_numerical': self.data.shape[1] - len(self.categorical_columns)
    }