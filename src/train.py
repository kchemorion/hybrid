import torch
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import wandb
import logging
from datetime import datetime
import sys
from typing import Dict, List, Optional, Tuple, Union

sys.path.append(str(Path(__file__).parent.parent))

from .models.hybrid import HybridImputationModel
from .data.loader import MissingValueDataset, create_dataloaders
from .data.preprocessor import (
    DataPreprocessor, 
    generate_mcar_mask, 
    generate_mar_mask, 
    generate_mnar_mask, 
    get_data_types
)
from .experiments.trainer import Trainer
from .utils.constants import ModelConfig, TrainingConfig, DataConfig

def setup_logging(config: TrainingConfig):
    """Setup logging configuration."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path: str):
    """Load your dataset. Adapt as needed."""
    try:
        if data_path.endswith(".csv"):
            data = pd.read_csv(data_path)  # Load from CSV
            data = data.values  # Convert to numpy array
        elif data_path.endswith(".npy"):
            data = np.load(data_path)  # Load from NumPy file
        else:
            raise ValueError("Unsupported data file format. Please provide a CSV or NumPy file.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Data file not found at {data_path}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading data: {e}")

def validate_config(config: dict) -> dict:
    """Validate and clean configuration."""
    required_sections = ['model', 'training', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model config
    required_model_params = {
        'model_type', 'hidden_dims', 'latent_dim', 'n_heads', 'dropout_rate',
        'batch_size', 'learning_rate', 'weight_decay'
    }
    missing_model_params = required_model_params - set(config['model'].keys())
    if missing_model_params:
        raise ValueError(f"Missing required model parameters: {missing_model_params}")
    
    # Validate data config
    required_data_params = {
        'data_path', 'missing_mechanism', 'missing_ratio', 'categorical_threshold',
        'seed'
    }
    missing_data_params = required_data_params - set(config['data'].keys())
    if missing_data_params:
        raise ValueError(f"Missing required data parameters: {missing_data_params}")
    
    # Validate training config
    required_training_params = {
        'device', 'n_epochs', 'early_stopping_patience', 'seed',
        'checkpoint_dir', 'log_dir', 'use_wandb'
    }
    missing_training_params = required_training_params - set(config['training'].keys())
    if missing_training_params:
        raise ValueError(f"Missing required training parameters: {missing_training_params}")
    
    return config

def main(config_path: str):
    """Main training function."""
    # Load and validate configuration
    config = load_config(config_path)
    config = validate_config(config)
    
    # Create config objects
    model_config = ModelConfig(**config['model'])
    training_config = TrainingConfig(**config['training'])
    data_config = DataConfig(**config['data'])
    
    # Setup logging
    logger = setup_logging(training_config)
    logger.info(f"Loading configuration from {config_path}")
    
    # Initialize wandb if enabled
    if training_config.use_wandb:
        wandb.init(
            project="missing_data_imputation",
            name=training_config.experiment_name,
            config=config
        )
    
    # Set device
    device = torch.device(training_config.device if torch.cuda.is_available() else "cpu")
    
    # Data Loading and Preprocessing
    logger.info("Loading and preprocessing data...")
    data = load_data(data_config.data_path)
    numerical_indices, categorical_indices = get_data_types(data, data_config.categorical_threshold)
    
    # Generate missing value mask based on specified mechanism
    if data_config.missing_mechanism.lower() == 'mcar':
        mask = generate_mcar_mask(data.shape, data_config.missing_ratio, data_config.seed)
    elif data_config.missing_mechanism.lower() == 'mar':
        mask = generate_mar_mask(data, data_config.missing_ratio, data_config.seed)
    elif data_config.missing_mechanism.lower() == 'mnar':
        mask = generate_mnar_mask(data, data_config.missing_ratio, data_config.seed)
    else:
        raise ValueError(f"Unknown missing mechanism: {data_config.missing_mechanism}")
    
    # Preprocess data
    data_preprocessor = DataPreprocessor(
        numerical_indices=numerical_indices,
        categorical_indices=categorical_indices,
        seed=training_config.seed
    )
    data, mask = data_preprocessor.fit_transform(data, mask)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data=data,
        mask=mask,
        batch_size=model_config.batch_size,
        seed=training_config.seed
    )
    
    # Create and initialize model
    logger.info("Initializing model...")
    input_dim = data.shape[1]
    model = HybridImputationModel(input_dim, model_config)
    model.to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_params={
            "lr": model_config.learning_rate,
            "weight_decay": model_config.weight_decay
        },
        device=device,
        checkpoint_dir=training_config.checkpoint_dir,
        use_wandb=training_config.use_wandb
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        n_epochs=training_config.n_epochs,
        early_stopping_patience=training_config.early_stopping_patience,
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train imputation model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    main(args.config)