# src/train.py

import torch
import argparse
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
import logging
from datetime import datetime
import sys
from typing import Dict, List, Optional, Tuple, Union

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae import VariationalAutoencoder
from src.models.hybrid import HybridImputationModel
from src.models.ensemble import EnsembleModel
from src.data.loader import MissingValueDataset
from src.data.preprocessor import DataPreprocessor
from src.experiments.trainer import Trainer
from src.experiments.evaluator import Evaluator
from src.utils.metrics import ImputationMetrics
from src.utils.visualization import ImputationVisualizer
from src.utils.constants import ModelConfig, ExperimentConfig, DataConfig

def setup_logging(config: ExperimentConfig):
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


def prepare_data(config: DataConfig) -> tuple:
    """Prepare datasets and dataloaders."""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        categorical_threshold=config.categorical_threshold,
        scaling_method=config.scaling_method,
        categorical_encoding=config.categorical_encoding
    )
    
    # Load and preprocess data
    if config.use_synthetic:
        from src.data.synthesizer import generate_mixed_type_data
        n_continuous = config.n_features - config.n_categorical
        data, mask = generate_mixed_type_data(
            n_samples=config.n_samples,
            n_continuous=n_continuous,
            n_categorical=config.n_categorical,
            missing_mechanism=config.missing_mechanism
        )
        input_dim = data.shape[1]  # Get input dimension from data
        preprocessor.input_dim = input_dim  # Store it in preprocessor
    else:
        raise NotImplementedError("Real data loading not implemented yet")
    
    # Create datasets
    train_dataset = MissingValueDataset(
        data=data,
        missing_mechanism=config.missing_mechanism,
        missing_ratio=config.missing_ratio,
        augmentation=config.use_augmentation
    )
    
    val_dataset = MissingValueDataset(
        data=data,
        missing_mechanism=config.missing_mechanism,
        missing_ratio=config.missing_ratio,
        augmentation=False
    )
    
    # Create dataloaders
    train_loader = train_dataset.get_dataloader(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader, preprocessor

def create_model(config: ModelConfig, input_dim: int):
    """Create model based on configuration."""
    if config.model_type == 'vae':
        model = VariationalAutoencoder(input_dim, config)
    elif config.model_type == 'hybrid':
        model = HybridImputationModel(input_dim, config)
    elif config.model_type == 'ensemble':
        model = EnsembleModel(input_dim, config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    return model

def main(config_path: str):
    """Main training function."""
    # Load and validate configuration
    logger = setup_logging(ExperimentConfig())
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        config = load_config(config_path)
        config = validate_config(config)
        
        model_config = ModelConfig(**config['model'])
        exp_config = ExperimentConfig(**config['experiment'])
        data_config = DataConfig(**config['data'])
        
    except Exception as e:
        logger.error(f"Error in configuration: {str(e)}")
        raise
    
    # Initialize wandb if enabled
    if exp_config.use_wandb:
        wandb.init(
            project="missing_data_imputation",
            name=exp_config.experiment_name,
            config=config
        )
    
    # Prepare data
    logger.info("Preparing data...")
    train_loader, val_loader, preprocessor = prepare_data(data_config)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(model_config, preprocessor.input_dim)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=exp_config.experiment_name,
        checkpoint_dir=exp_config.checkpoint_dir,
        use_wandb=exp_config.use_wandb,
        device=device
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        n_epochs=model_config.n_epochs,
        early_stopping_patience=model_config.early_stopping_patience,
        gradient_clip_val=model_config.gradient_clip_val
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = Evaluator(
        true_data=val_loader.dataset.data,
        mask=val_loader.dataset.mask,
        feature_names=preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else None,
        save_dir=exp_config.results_dir
    )
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        imputed_data, uncertainties = model.impute(
            val_loader.dataset.data.to(device),
            val_loader.dataset.mask.to(device),
            return_uncertainties=True
        )
    
    # Move tensors back to CPU for evaluation
    imputed_data = imputed_data.cpu()
    uncertainties = uncertainties.cpu()
    
    # Evaluate and visualize results
    evaluator.evaluate_model(
        model_name=exp_config.experiment_name,
        imputed_data=imputed_data,
        uncertainties=uncertainties
    )
    
    # Generate visualization
    visualizer = ImputationVisualizer(save_dir=exp_config.results_dir)
    visualizer.plot_imputation_results(
        true_values=val_loader.dataset.data,
        imputed_values=imputed_data,
        uncertainties=uncertainties,
        feature_names=preprocessor.feature_names if hasattr(preprocessor, 'feature_names') else None,
        title=exp_config.experiment_name
    )
    
    logger.info("Experiment completed successfully!")

def validate_config(config: dict) -> dict:
    """Validate and clean configuration."""
    required_sections = ['model', 'experiment', 'data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model config
    required_model_params = {
        'model_type', 'hidden_dims', 'latent_dim', 'n_heads', 'dropout_rate',
        'batch_size', 'learning_rate', 'weight_decay', 'n_epochs'
    }
    missing_model_params = required_model_params - set(config['model'].keys())
    if missing_model_params:
        raise ValueError(f"Missing required model parameters: {missing_model_params}")
    
    # Validate data config
    required_data_params = {
        'missing_mechanism', 'missing_ratio', 'categorical_threshold',
        'scaling_method', 'categorical_encoding', 'use_augmentation',
        'noise_std', 'use_synthetic', 'n_samples', 'n_features',
        'n_categorical', 'correlation_strength'
    }
    missing_data_params = required_data_params - set(config['data'].keys())
    if missing_data_params:
        raise ValueError(f"Missing required data parameters: {missing_data_params}")
    
    # Validate experiment config
    required_exp_params = {
        'experiment_name', 'experiment_version', 'data_dir', 'checkpoint_dir',
        'results_dir', 'log_dir', 'use_wandb', 'log_interval', 'save_interval',
        'device', 'num_workers', 'pin_memory'
    }
    missing_exp_params = required_exp_params - set(config['experiment'].keys())
    if missing_exp_params:
        raise ValueError(f"Missing required experiment parameters: {missing_exp_params}")
    
    return config

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
