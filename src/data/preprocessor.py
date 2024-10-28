import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

class DataPreprocessor:
    """Data preprocessing for missing value imputation."""

    def __init__(
        self,
        categorical_threshold: int = 10,
        scaling_method: str = "standard",
        categorical_encoding: str = "onehot",
        initial_imputation: str = "mean",
    ):
        self.categorical_threshold = categorical_threshold
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.initial_imputation = initial_imputation

        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_types: Dict[str, str] = {}
        self.feature_names = []

    def fit(self, data: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> "DataPreprocessor":
        """Fit preprocessor to data."""
        self.feature_names = data.columns.tolist()
        self.feature_types = self._detect_feature_types(data, categorical_columns)
        self._fit_imputers(data)
        self._fit_encoders(data)
        self._fit_scalers(data)
        return self

    def transform(self, data: pd.DataFrame, return_mask: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform data."""
        data_copy = data.copy()
        mask = ~data_copy.isna()
        data_copy = self._initial_impute(data_copy)
        data_copy = self._transform_categorical(data_copy)
        data_copy = self._transform_numerical(data_copy)
        if return_mask:
            return data_copy.values, mask.values
        return data_copy.values

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Inverse transform data."""
        df = pd.DataFrame(data, columns=self.feature_names)
        for col in self.feature_types:
            if self.feature_types[col] == "numerical":
                df[col] = self.scalers[col].inverse_transform(df[[col]])
            elif self.feature_types[col] == "categorical":
                df[col] = self.encoders[col].inverse_transform(df[col].astype(int))
        return df

    def _detect_feature_types(self, data: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> Dict[str, str]:
        """Detect feature types."""
        feature_types = {}
        for col in data.columns:
            if categorical_columns and col in categorical_columns:
                feature_types[col] = "categorical"
            elif data[col].dtype in ["object", "category"] or len(data[col].unique()) < self.categorical_threshold:
                feature_types[col] = "categorical"
            else:
                feature_types[col] = "numerical"
        return feature_types

    def _fit_imputers(self, data: pd.DataFrame):
        """Fit imputers for missing values."""
        for col in self.feature_types:
            if data[col].isna().any():
                if self.feature_types[col] == "numerical":
                    imputer = SimpleImputer(strategy=self.initial_imputation)
                else:
                    imputer = SimpleImputer(strategy="most_frequent")
                imputer.fit(data[[col]])
                self.imputers[col] = imputer

    def _fit_encoders(self, data: pd.DataFrame):
        """Fit encoders for categorical variables."""
        for col in self.feature_types:
            if self.feature_types[col] == "categorical":
                encoder = LabelEncoder()
                encoder.fit(data[col].fillna("MISSING"))
                self.encoders[col] = encoder

    def _fit_scalers(self, data: pd.DataFrame):
        """Fit scalers for numerical variables."""
        for col in self.feature_types:
            if self.feature_types[col] == "numerical":
                scaler = StandardScaler() if self.scaling_method == "standard" else MinMaxScaler()
                scaler.fit(data[[col]].dropna().values.reshape(-1, 1))
                self.scalers[col] = scaler

    def _initial_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform initial imputation."""
        for col in self.feature_types:
            if data[col].isna().any():
                data[col] = self.imputers[col].transform(data[[col]])
        return data

    def _transform_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables."""
        for col in self.feature_types:
            if self.feature_types[col] == "categorical":
                data[col] = self.encoders[col].transform(data[col])
        return data

    def _transform_numerical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical variables."""
        for col in self.feature_types:
            if self.feature_types[col] == "numerical":
                data[col] = self.scalers[col].transform(data[[col]])
        return data

    def get_feature_info(self) -> Dict[str, Dict]:
        """Get feature information."""
        return {
            "types": self.feature_types,
            "encoders": {col: type(enc).__name__ for col, enc in self.encoders.items()},
            "scalers": {col: type(scl).__name__ for col, scl in self.scalers.items()},
        }