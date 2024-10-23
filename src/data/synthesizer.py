# src/data/synthesizer.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.distributions import Normal, Bernoulli

@dataclass
class ClinicalTrialParams:
    """Parameters from the clinical trial for data synthesis."""
    n_samples: int = 404
    n_timepoints: int = 24
    treatment_groups: List[str] = ('Rexlemestrocel-L', 'Rexlemestrocel-L + HA', 'Placebo')
    baseline_demographics: Dict = None
    treatment_effects: Dict = None
    missing_patterns: Dict = None
    outcome_correlations: Dict = None

class ClinicalTrialSynthesizer:
    """
    Generates synthetic clinical trial data based on real trial parameters.
    
    Features:
    - Realistic patient demographics
    - Treatment effect simulation
    - Temporal correlation structure
    - Multiple outcome measures
    - Missing data patterns
    """
    
    def __init__(
        self,
        params: ClinicalTrialParams,
        random_seed: Optional[int] = None
    ):
        self.params = params
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
        # Initialize default parameters if not provided
        self._initialize_default_params()
        
    def _initialize_default_params(self):
        """Initialize default parameters based on the clinical trial."""
        if self.params.baseline_demographics is None:
            self.params.baseline_demographics = {
                'age': {'mean': 42.8, 'std': 10.96},
                'gender_ratio': 0.433,  # female proportion
                'weight': {'mean': 75.0, 'std': 15.0}
            }
            
        if self.params.treatment_effects is None:
            # Treatment effects from trial
            self.params.treatment_effects = {
                'Rexlemestrocel-L': {
                    'vas_pain': -0.267,
                    'odi_score': -0.378
                },
                'Rexlemestrocel-L + HA': {
                    'vas_pain': -0.335,
                    'odi_score': -0.409
                },
                'Placebo': {
                    'vas_pain': -0.313,
                    'odi_score': -0.413
                }
            }
            
        if self.params.missing_patterns is None:
            self.params.missing_patterns = {
                'dropout_rate': 0.15,
                'missing_visit_prob': 0.05,
                'incomplete_data_prob': 0.1
            }
            
    def generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic clinical trial data.
        
        Returns:
            Tuple of (patient_data, longitudinal_data)
        """
        # Generate patient baseline data
        patient_data = self._generate_patient_data()
        
        # Generate longitudinal outcomes
        longitudinal_data = self._generate_longitudinal_data(patient_data)
        
        # Apply missing data patterns
        longitudinal_data = self._apply_missing_patterns(longitudinal_data)
        
        return patient_data, longitudinal_data
    
    def _generate_patient_data(self) -> pd.DataFrame:
        """Generate baseline patient characteristics."""
        n_samples = self.params.n_samples
        
        # Generate demographics
        ages = stats.norm.rvs(
            loc=self.params.baseline_demographics['age']['mean'],
            scale=self.params.baseline_demographics['age']['std'],
            size=n_samples
        )
        
        gender = stats.bernoulli.rvs(
            p=self.params.baseline_demographics['gender_ratio'],
            size=n_samples
        )
        
        weights = stats.norm.rvs(
            loc=self.params.baseline_demographics['weight']['mean'],
            scale=self.params.baseline_demographics['weight']['std'],
            size=n_samples
        )
        
        # Assign treatments
        treatments = np.random.choice(
            self.params.treatment_groups,
            size=n_samples,
            p=[1/3, 1/3, 1/3]
        )
        
        # Create DataFrame
        patient_data = pd.DataFrame({
            'patient_id': range(n_samples),
            'age': ages,
            'gender': gender,
            'weight': weights,
            'treatment': treatments
        })
        
        return patient_data
    
    def _generate_longitudinal_data(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """Generate longitudinal outcome data."""
        records = []
        
        for _, patient in patient_data.iterrows():
            patient_outcomes = self._generate_patient_outcomes(patient)
            records.extend(patient_outcomes)
            
        return pd.DataFrame(records)
    
    def _generate_patient_outcomes(self, patient: pd.Series) -> List[Dict]:
        """Generate longitudinal outcomes for a single patient."""
        treatment_effect = self.params.treatment_effects[patient['treatment']]
        records = []
        
        # Generate baseline values
        baseline_vas = stats.norm.rvs(loc=70, scale=15)
        baseline_odi = stats.norm.rvs(loc=50, scale=10)
        
        # Generate temporal correlation
        temporal_noise = stats.multivariate_normal.rvs(
            mean=np.zeros(self.params.n_timepoints),
            cov=self._generate_temporal_correlation_matrix()
        )
        
        for timepoint in range(self.params.n_timepoints):
            # Apply treatment effect with temporal progression
            effect_multiplier = 1 - np.exp(-timepoint / 12)  # Exponential approach to max effect
            
            vas_pain = baseline_vas + (
                treatment_effect['vas_pain'] * effect_multiplier * 100 +
                temporal_noise[timepoint] * 10
            )
            
            odi_score = baseline_odi + (
                treatment_effect['odi_score'] * effect_multiplier * 100 +
                temporal_noise[timepoint] * 8
            )
            
            records.append({
                'patient_id': patient['patient_id'],
                'timepoint': timepoint,
                'vas_pain': max(0, min(100, vas_pain)),
                'odi_score': max(0, min(100, odi_score))
            })
            
        return records
    
# src/data/synthesizer.py (continued)

    def _generate_temporal_correlation_matrix(self) -> np.ndarray:
        """Generate temporal correlation matrix for outcomes."""
        times = np.arange(self.params.n_timepoints)
        distance_matrix = np.abs(times[:, np.newaxis] - times)
        
        # Exponential decay correlation
        correlation_matrix = np.exp(-distance_matrix / 6)  # 6-month correlation length
        
        # Ensure positive definiteness
        correlation_matrix += np.eye(self.params.n_timepoints) * 1e-6
        
        return correlation_matrix
    
    def _apply_missing_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply realistic missing data patterns to the longitudinal data."""
        data = data.copy()
        
        # Apply dropout
        patient_ids = data['patient_id'].unique()
        dropout_patients = np.random.choice(
            patient_ids,
            size=int(len(patient_ids) * self.params.missing_patterns['dropout_rate']),
            replace=False
        )
        
        for patient_id in dropout_patients:
            dropout_time = np.random.randint(1, self.params.n_timepoints)
            data.loc[
                (data['patient_id'] == patient_id) & 
                (data['timepoint'] >= dropout_time),
                ['vas_pain', 'odi_score']
            ] = np.nan
            
        # Apply missing visits
        mask = np.random.random(len(data)) < self.params.missing_patterns['missing_visit_prob']
        data.loc[mask, ['vas_pain', 'odi_score']] = np.nan
        
        # Apply incomplete data
        mask = np.random.random(len(data)) < self.params.missing_patterns['incomplete_data_prob']
        data.loc[mask, 'vas_pain'] = np.nan
        
        mask = np.random.random(len(data)) < self.params.missing_patterns['incomplete_data_prob']
        data.loc[mask, 'odi_score'] = np.nan
        
        return data

class SyntheticDataGenerator:
    """
    Advanced synthetic data generator with multiple simulation capabilities.
    
    Features:
    - Multiple missing data mechanisms
    - Complex correlation structures
    - Non-linear relationships
    - Time-varying effects
    - Batch generation
    """
    
    def __init__(
        self,
        n_features: int,
        n_samples: int,
        missing_mechanism: str = 'MCAR',
        correlation_strength: float = 0.5,
        nonlinearity: str = 'moderate',
        random_seed: Optional[int] = None
    ):
        self.n_features = n_features
        self.n_samples = n_samples
        self.missing_mechanism = missing_mechanism
        self.correlation_strength = correlation_strength
        self.nonlinearity = nonlinearity
        
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            
        # Generate correlation matrix
        self.correlation_matrix = self._generate_correlation_matrix()
        
    def _generate_correlation_matrix(self) -> np.ndarray:
        """Generate feature correlation matrix."""
        if self.correlation_strength == 0:
            return np.eye(self.n_features)
            
        # Generate base correlation matrix
        random_matrix = np.random.randn(self.n_features, self.n_features)
        base_correlation = random_matrix @ random_matrix.T
        
        # Scale correlations
        correlation_matrix = (
            (base_correlation / np.max(np.abs(base_correlation))) * 
            self.correlation_strength
        )
        
        # Ensure positive definiteness
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        correlation_matrix[np.diag_indices_from(correlation_matrix)] = 1
        
        # Add small positive constant to ensure positive definiteness
        min_eigenval = np.min(np.linalg.eigvals(correlation_matrix))
        if min_eigenval < 0:
            correlation_matrix += (-min_eigenval + 1e-6) * np.eye(self.n_features)
            
        return correlation_matrix
    
    def generate_batch(
        self,
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of synthetic data with missing values."""
        if batch_size is None:
            batch_size = self.n_samples
            
        # Generate correlated features
        data = stats.multivariate_normal.rvs(
            mean=np.zeros(self.n_features),
            cov=self.correlation_matrix,
            size=batch_size
        )
        
        # Apply non-linear transformations
        data = self._apply_nonlinearity(data)
        
        # Generate missing value mask
        mask = self._generate_missing_mask(data)
        
        return torch.FloatTensor(data), torch.FloatTensor(mask)
    
    def _apply_nonlinearity(self, data: np.ndarray) -> np.ndarray:
        """Apply non-linear transformations to features."""
        if self.nonlinearity == 'none':
            return data
            
        transformed_data = data.copy()
        
        if self.nonlinearity == 'moderate':
            # Apply moderate non-linear transformations
            for i in range(self.n_features):
                if i % 3 == 0:
                    transformed_data[:, i] = np.sin(data[:, i])
                elif i % 3 == 1:
                    transformed_data[:, i] = np.exp(data[:, i] / 2)
                    
        elif self.nonlinearity == 'strong':
            # Apply stronger non-linear transformations
            for i in range(self.n_features):
                if i % 4 == 0:
                    transformed_data[:, i] = np.sin(data[:, i] * 2)
                elif i % 4 == 1:
                    transformed_data[:, i] = np.exp(data[:, i])
                elif i % 4 == 2:
                    transformed_data[:, i] = data[:, i]**3
                    
        # Standardize transformed data
        transformed_data = (transformed_data - np.mean(transformed_data, axis=0)) / np.std(transformed_data, axis=0)
        
        return transformed_data
    
    def _generate_missing_mask(self, data: np.ndarray) -> np.ndarray:
        """Generate missing value mask based on specified mechanism."""
        mask = np.ones_like(data)
        missing_ratio = 0.2
        
        if self.missing_mechanism == 'MCAR':
            mask = np.random.random(data.shape) > missing_ratio
            
        elif self.missing_mechanism == 'MAR':
            for j in range(self.n_features):
                # Make missingness depend on other features
                predictor_idx = (j + 1) % self.n_features
                predictor_values = data[:, predictor_idx]
                missing_prob = stats.norm.cdf((predictor_values - np.mean(predictor_values)) / np.std(predictor_values))
                mask[:, j] = np.random.random(len(data)) > (missing_prob * missing_ratio)
                
        elif self.missing_mechanism == 'MNAR':
            for j in range(self.n_features):
                # Make missingness depend on the values themselves
                values = data[:, j]
                missing_prob = stats.norm.cdf((values - np.mean(values)) / np.std(values))
                mask[:, j] = np.random.random(len(data)) > (missing_prob * missing_ratio)
                
        return mask


# src/data/synthesizer.py

def generate_mixed_type_data(
    n_samples: int,
    n_continuous: int,
    n_categorical: int,
    missing_mechanism: str = 'MCAR'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate mixed-type data with specified characteristics."""
    # Generate continuous features
    continuous_data = np.random.randn(n_samples, n_continuous)
    
    # Generate categorical features
    categorical_data = np.random.randint(
        0, 3,  # 3 categories
        size=(n_samples, n_categorical)
    )
    
    # Combine all features
    data = np.hstack([continuous_data, categorical_data])
    
    # Create feature names
    columns = (
        [f'continuous_{i}' for i in range(n_continuous)] +
        [f'categorical_{i}' for i in range(n_categorical)]
    )
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Generate missing mask
    mask = np.random.rand(*data.shape) > 0.2  # 20% missing values
    mask_df = pd.DataFrame(mask, columns=columns)
    
    return df, mask_df