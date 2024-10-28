# src/data/preprocessor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Optional

class DataPreprocessor:

    def __init__(self, numerical_indices: List[int], categorical_indices: List[int], seed: int = 42):
        self.numerical_indices = numerical_indices
        self.categorical_indices = categorical_indices
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.imputer_numerical = SimpleImputer(strategy="mean")
        self.imputer_categorical = SimpleImputer(strategy="most_frequent")
        self.seed = seed
        np.random.seed(self.seed)


        self.feature_names = []

    def fit_transform(self, data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if isinstance(data, pd.DataFrame):
             self.feature_names = list(data.columns)

        # Separate numerical and categorical features
        data_numerical = data[:, self.numerical_indices]
        data_categorical = data[:, self.categorical_indices]

        # Impute missing values in both feature types before scaling/encoding.
        data_numerical = self.imputer_numerical.fit_transform(data_numerical)
        data_categorical = self.imputer_categorical.fit_transform(data_categorical)
        # Scale numerical features
        data_numerical = self.numerical_scaler.fit_transform(data_numerical)

        # One-hot encode categorical features
        data_categorical = self.categorical_encoder.fit_transform(data_categorical)

        # Combine preprocessed features
        data_processed = np.concatenate([data_numerical, data_categorical], axis=1)

        #Apply same imputation to masks
        mask_num = self.imputer_numerical.transform(mask[:, self.numerical_indices])
        mask_num = self.numerical_scaler.transform(mask_num)

        #For categorical masks we use one hot encoding

        mask_cat = self.categorical_encoder.transform(mask[:, self.categorical_indices])


        mask_processed = np.concatenate([mask_num, mask_cat], axis=1)

        return data_processed, mask_processed
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray: # Changed to return a numpy array since original data may be numpy array

        """ Inverse Transform the data"""

        #Get numerical and categorical feature counts from fitted encoders/scalers
        n_num_features = (
            self.numerical_scaler.n_features_in_
            if hasattr(self.numerical_scaler, "n_features_in_")
            else len(self.numerical_indices) #for compatibility with older scikit-learn versions
        )
        
        n_cat_features = (
            self.categorical_encoder.n_features_in_
            if hasattr(self.categorical_encoder, "n_features_in_")
            else self.categorical_encoder.categories_[0].shape[0] # for compatability with older scikit-learn versions
        )

        # Numerical features come first after preprocessing
        data_numerical = data[:, :n_num_features]
        data_categorical = data[:, n_num_features:]

        #Inverse transform numerical features

        data_numerical = self.numerical_scaler.inverse_transform(data_numerical)
        data_numerical = self.imputer_numerical.inverse_transform(data_numerical)  #Apply numerical imputer


        #Inverse transform categorical features
        data_categorical = self.categorical_encoder.inverse_transform(data_categorical)
        #Apply categorical imputer
        data_categorical = self.imputer_categorical.inverse_transform(data_categorical)


        #Combine the data
        data_processed = np.concatenate([data_numerical, data_categorical], axis=1) #Modified to work with numpy arrays.



        return data_processed