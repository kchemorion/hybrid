# src/experiments/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Callable
import wandb
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json

from ..models.base import BaseModel
from ..utils.metrics import ImputationMetrics

class Trainer:
    """
    Advanced training framework for missing data imputation models.
    
    Features:
    - Multiple loss functions
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Experiment tracking
    - Model checkpointing
    - Validation monitoring
    """
    
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_params: Dict = None,
        scheduler_class: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_params: Dict = None,
        loss_weights: Dict[str, float] = None,
        device: str = None,
        experiment_name: str = "imputation_experiment",
        checkpoint_dir: str = "./checkpoints",
        use_wandb: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        optimizer_params = optimizer_params or {'lr': 1e-3}
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_params
        )
        
        # Initialize scheduler
        self.scheduler = None
        if scheduler_class:
            scheduler_params = scheduler_params or {}
            self.scheduler = scheduler_class(
                self.optimizer,
                **scheduler_params
            )
            
        # Initialize loss weights
        self.loss_weights = loss_weights or {
            'reconstruction': 1.0,
            'kl_divergence': 0.1,
            'uncertainty': 0.1
        }
        
        # Setup logging and tracking
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="missing_data_imputation",
                name=experiment_name,
                config={
                    "model_config": model.config,
                    "optimizer": optimizer_class.__name__,
                    "optimizer_params": optimizer_params,
                    "scheduler": scheduler_class.__name__ if scheduler_class else None,
                    "loss_weights": self.loss_weights
                }
            )
            
        # Setup metrics
        self.metrics = ImputationMetrics()
        
        # Initialize best validation score for model selection
        self.best_val_score = float('inf')
        self.best_epoch = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def train(
        self,
        n_epochs: int,
        early_stopping_patience: int = 10,
        gradient_clip_val: float = 1.0
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            n_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement
            gradient_clip_val: Maximum gradient norm
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': []
        }
        
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training phase
            train_metrics = self._train_epoch(gradient_clip_val)
            history['train_loss'].append(train_metrics['total_loss'])
            
            # Validation phase
            if self.val_loader:
                val_metrics = self._validate_epoch()
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_rmse'].append(val_metrics['rmse'])
                history['val_mae'].append(val_metrics['mae'])
                
                # Model selection and early stopping
                if val_metrics['rmse'] < self.best_val_score:
                    self.best_val_score = val_metrics['rmse']
                    self.best_epoch = epoch
                    patience_counter = 0
                    self._save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                    
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics if self.val_loader else None)
            
        # Load best model
        if self.val_loader:
            self._load_checkpoint('best_model.pt')
            
        return history
    
    # src/experiments/trainer.py

    def _train_epoch(self, gradient_clip_val: float) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            x, mask = batch
            x = x.to(self.device).float()
            mask = mask.to(self.device).bool()  # Explicitly convert to boolean
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x, mask)
            losses = self.model.compute_loss(x, mask, outputs)
            
            # Make sure total_loss is a tensor
            total_loss = losses['total_loss']
            if not isinstance(total_loss, torch.Tensor):
                raise ValueError(f"Expected total_loss to be a tensor, got {type(total_loss)}")
            
            # Compute weighted loss
            weighted_loss = sum(
                self.loss_weights.get(k, 1.0) * v 
                for k, v in losses.items() 
                if k in self.loss_weights and isinstance(v, torch.Tensor)
            )
            
            # Ensure weighted_loss is a float
            weighted_loss = float(weighted_loss)
            weighted_loss = torch.tensor(weighted_loss, dtype=torch.float32, requires_grad=True, device=self.device)
            
            # Backward pass
            weighted_loss.backward()
            
            # Gradient clipping
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    gradient_clip_val
                )
                
            self.optimizer.step()
            
            # Convert loss values to float for metrics
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in losses.items()}
            epoch_metrics.append(metrics)
            
        # Average metrics over epoch
        avg_metrics = {}
        for k in epoch_metrics[0].keys():
            values = [m[k] for m in epoch_metrics]
            avg_metrics[k] = np.mean(values)
            
        return avg_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, mask = batch
                x = x.to(self.device)
                mask = mask.to(self.device)
                
                # Convert mask to boolean type explicitly
                mask = mask.bool()
                
                # Forward pass
                outputs = self.model(x, mask)
                losses = self.model.compute_loss(x, mask, outputs)
                
                # Compute metrics on masked elements
                masked_indices = ~mask  # Now this operation will work correctly
                metrics = self.metrics.compute_metrics(
                    x[masked_indices],
                    outputs['imputed'][masked_indices]
                )
                metrics.update(losses)
                epoch_metrics.append(metrics)
        
        # Average metrics over epoch
        avg_metrics = {}
        for k in epoch_metrics[0].keys():
            values = [m[k] for m in epoch_metrics]
            avg_metrics[k] = np.mean([v.item() if torch.is_tensor(v) else v for v in values])
        
        return avg_metrics
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log metrics to console and wandb."""
        metrics_str = f"Epoch {epoch}: "
        metrics_dict = {}
        
        # Format training metrics
        for k, v in train_metrics.items():
            metrics_str += f"train_{k}: {v:.4f} "
            metrics_dict[f"train_{k}"] = v
            
        # Format validation metrics
        if val_metrics:
            for k, v in val_metrics.items():
                metrics_str += f"val_{k}: {v:.4f} "
                metrics_dict[f"val_{k}"] = v
                
        self.logger.info(metrics_str)
        
        if self.use_wandb:
            wandb.log(metrics_dict, step=epoch)
            
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'best_epoch': self.best_epoch
        }
        
        torch.save(
            checkpoint,
            self.checkpoint_dir / filename
        )
        
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.best_val_score = checkpoint['best_val_score']
        self.best_epoch = checkpoint['best_epoch']
        
    def test(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        if not self.test_loader:
            raise ValueError("Test loader not provided")
            
        self.model.eval()
        test_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                x, mask = batch
                x, mask = x.to(self.device), mask.to(self.device)
                
                # Forward pass
                outputs = self.model(x, mask)
                
                # Compute metrics
                metrics = self.metrics.compute_metrics(
                    x[~mask],
                    outputs['imputed'][~mask]
                )
                test_metrics.append(metrics)
                
        # Average metrics
        avg_metrics = {
            k: np.mean([m[k].item() for m in test_metrics])
            for k in test_metrics[0].keys()
        }
        
        # Log test metrics
        metrics_str = "Test metrics: "
        for k, v in avg_metrics.items():
            metrics_str += f"{k}: {v:.4f} "
        self.logger.info(metrics_str)
        
        if self.use_wandb:
            wandb.log({f"test_{k}": v for k, v in avg_metrics.items()})
            
        return avg_metrics

