# src/utils/constants.py

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    # Model Type
    model_type: str = "hybrid"  # Added this field
    
    # Model Architecture
    input_dim: int = None
    hidden_dims: List[int] = None
    latent_dim: int = 32
    n_heads: int = 4
    dropout_rate: float = 0.1
    
    # Training Parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    
    # Loss Weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    uncertainty_weight: float = 0.1
    
    # Scheduler Parameters
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-6
    
    # Validation
    validation_split: float = 0.2
    test_split: float = 0.1

    def __post_init__(self):
        """Set default hidden_dims if None"""
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]

@dataclass
class ExperimentConfig:
    """Configuration for experiment running and logging."""
    
    # Experiment Identification
    experiment_name: str = "imputation_experiment"
    experiment_version: str = "v1.0"
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    log_dir: str = "./logs"
    
    # Logging
    use_wandb: bool = True
    log_interval: int = 100
    save_interval: int = 1000
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class DataConfig:
    """Configuration for data processing and augmentation."""
    
    # Missing Data Generation
    missing_mechanism: str = "MCAR"
    missing_ratio: float = 0.2
    
    # Data Processing
    categorical_threshold: int = 10
    scaling_method: str = "standard"
    categorical_encoding: str = "onehot"
    
    # Augmentation
    use_augmentation: bool = True
    noise_std: float = 0.1
    
    # Synthetic Data Generation
    use_synthetic: bool = True  # Added this field
    n_samples: int = 10000
    n_features: int = 20
    n_categorical: int = 5
    correlation_strength: float = 0.5
    
    # Batch Processing
    batch_size: int = 64  # Added if needed
    num_workers: int = 4  # Added if needed
    pin_memory: bool = True  # Added if needed

# Default configurations
DEFAULT_MODEL_CONFIG = {
    'hidden_dims': [256, 128, 64],
    'latent_dim': 32,
    'n_heads': 4,
    'dropout_rate': 0.1,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'n_epochs': 100
}

DEFAULT_TRAINING_CONFIG = {
    'early_stopping_patience': 10,
    'gradient_clip_val': 1.0,
    'validation_split': 0.2,
    'test_split': 0.1
}

DEFAULT_LOSS_WEIGHTS = {
    'reconstruction': 1.0,
    'kl_divergence': 0.1,
    'uncertainty': 0.1,
    'diversity': 0.05
}

# Error messages
ERROR_MESSAGES = {
    'data_not_found': "Data file not found at specified path",
    'invalid_config': "Invalid configuration provided",
    'training_error': "Error occurred during training",
    'evaluation_error': "Error occurred during evaluation",
    'model_save_error': "Error saving model checkpoint",
    'model_load_error': "Error loading model checkpoint"
}

# Metric names
METRIC_NAMES = {
    'rmse': 'Root Mean Square Error',
    'mae': 'Mean Absolute Error',
    'mape': 'Mean Absolute Percentage Error',
    'r2': 'R-squared Score',
    'correlation': 'Pearson Correlation',
    'calibration_error': 'Calibration Error'
}