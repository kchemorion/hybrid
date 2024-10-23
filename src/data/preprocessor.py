# src/data/preprocessor.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

# src/data/preprocessor.py

class DataPreprocessor:
    """
    Comprehensive data preprocessing for missing value imputation.
    """
    
# src/data/preprocessor.py

class DataPreprocessor:
    def __init__(
        self,
        categorical_threshold: int = 10,
        scaling_method: str = 'standard',
        categorical_encoding: str = 'onehot'
    ):
        self.categorical_threshold = categorical_threshold
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        

        
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_types: Dict[str, str] = {}
        self.statistics: Dict[str, Dict] = {}
    
    def identify_categorical_columns(self, data: pd.DataFrame) -> List[int]:
        """Identify categorical columns based on threshold and data type."""
        categorical_cols = []
        for i, col in enumerate(data.columns):
            if (data[col].dtype == 'object' or 
                data[col].dtype == 'category' or 
                len(data[col].unique()) < self.categorical_threshold):
                categorical_cols.append(i)
        return categorical_cols
        
    def fit(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> 'DataPreprocessor':
        """
        Fit preprocessor to data.
        
        Args:
            data: Input DataFrame
            categorical_columns: List of categorical column names
        """
        self.columns = data.columns.tolist()
        
        # Detect feature types if not provided
        self.feature_types = self._detect_feature_types(
            data, 
            categorical_columns
        )
        
        # Calculate and store statistics
        self._calculate_statistics(data)
        
        # Fit encoders for categorical variables
        self._fit_categorical_encoders(data)
        
        # Fit scalers for numerical variables
        self._fit_numerical_scalers(data)
        
        return self
    
    def transform(
        self,
        data: pd.DataFrame,
        return_mask: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: Input DataFrame
            return_mask: Whether to return missing value mask
            
        Returns:
            Transformed data array and optionally missing value mask
        """
        # Create copy to avoid modifying original data
        data_copy = data.copy()
        
        # Generate missing value mask
        mask = ~data_copy.isna()
        
        # Initial imputation
        data_copy = self._initial_impute(data_copy)
        
        # Transform categorical variables
        data_copy = self._transform_categorical(data_copy)
        
        # Scale numerical variables
        data_copy = self._transform_numerical(data_copy)
        
        if return_mask:
            return data_copy.values, mask.values
        return data_copy.values
    
    def inverse_transform(
        self,
        data: np.ndarray,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Inverse transform data back to original space.
        
        Args:
            data: Transformed data array
            columns: Column names (uses fitted columns if None)
            
        Returns:
            DataFrame in original space
        """
        if columns is None:
            columns = self.columns
            
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Inverse transform numerical variables
        for col in self.feature_types:
            if self.feature_types[col] == 'numerical':
                df[col] = self.scalers[col].inverse_transform(
                    df[[col]]
                )
        
        # Inverse transform categorical variables
        for col in self.feature_types:
            if self.feature_types[col] == 'categorical':
                df[col] = self.encoders[col].inverse_transform(
                    df[col].astype(int)
                )
                
        return df
    
    def _detect_feature_types(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Automatically detect feature types."""
        feature_types = {}
        
        for column in data.columns:
            if categorical_columns and column in categorical_columns:
                feature_types[column] = 'categorical'
            elif data[column].dtype in ['object', 'category']:
                feature_types[column] = 'categorical'
            elif len(data[column].unique()) < self.categorical_threshold:
                feature_types[column] = 'categorical'
            else:
                feature_types[column] = 'numerical'
                
        return feature_types
    
    def _calculate_statistics(self, data: pd.DataFrame):
        """Calculate and store relevant statistics for each feature."""
        for column in data.columns:
            self.statistics[column] = {
                'mean': data[column].mean() if self.feature_types[column] == 'numerical' else None,
                'std': data[column].std() if self.feature_types[column] == 'numerical' else None,
                'median': data[column].median() if self.feature_types[column] == 'numerical' else None,
                'mode': data[column].mode().iloc[0],
                'missing_ratio': data[column].isna().mean(),
                'unique_values': len(data[column].unique()),
                'correlations': data[column].corr() if self.feature_types[column] == 'numerical' else None
            }
    
    def _fit_categorical_encoders(self, data: pd.DataFrame):
        """Fit encoders for categorical variables."""
        for column in self.feature_types:
            if self.feature_types[column] == 'categorical':
                if self.categorical_encoder == 'label':
                    encoder = LabelEncoder()
                    # Fit encoder including NaN if present
                    non_null_data = data[column].fillna('MISSING')
                    encoder.fit(non_null_data)
                    self.encoders[column] = encoder
                    
    def _fit_numerical_scalers(self, data: pd.DataFrame):
        """Fit scalers for numerical variables."""
        for column in self.feature_types:
            if self.feature_types[column] == 'numerical':
                scaler = StandardScaler() if self.scaling_method == 'standard' else MinMaxScaler()
                # Fit scaler on non-missing values
                non_null_data = data[column].dropna().values.reshape(-1, 1)
                scaler.fit(non_null_data)
                self.scalers[column] = scaler
                
    def _initial_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform initial imputation for missing values."""
        for column in data.columns:
            if data[column].isna().any():
                if self.feature_types[column] == 'numerical':
                    if self.initial_imputation == 'mean':
                        value = self.statistics[column]['mean']
                    elif self.initial_imputation == 'median':
                        value = self.statistics[column]['median']
                    else:
                        value = 0
                else:
                    value = self.statistics[column]['mode']
                    
                data[column].fillna(value, inplace=True)
                
        return data
    
    def _transform_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables."""
        data_copy = data.copy()
        
        for column in self.feature_types:
            if self.feature_types[column] == 'categorical':
                if self.categorical_encoder == 'label':
                    data_copy[column] = self.encoders[column].transform(
                        data_copy[column].fillna('MISSING')
                    )
                    
        return data_copy
    
    def _transform_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical variables."""
        data_copy = data.copy()
        
        for column in self.feature_types:
            if self.feature_types[column] == 'numerical':
                data_copy[column] = self.scalers[column].transform(
                    data_copy[[column]]
                )
                
        return data_copy
    
    def get_feature_info(self) -> Dict[str, Dict]:
        """Get comprehensive feature information."""
        return {
            'types': self.feature_types,
            'statistics': self.statistics,
            'encoders': {col: type(enc).__name__ for col, enc in self.encoders.items()},
            'scalers': {col: type(scl).__name__ for col, scl in self.scalers.items()}
        }