# src/models/ensemble.py

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .base import BaseModel
from .vae import VariationalAutoencoder
from .hybrid import HybridImputationModel
import numpy as np

class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple imputation models.
    
    Features:
    - Multiple model types (VAE, Hybrid)
    - Weighted averaging based on model uncertainty
    - Bootstrapped training for diversity
    - Bayesian model averaging
    """
    
    def __init__(self, input_dim: int, config: Dict):
        super().__init__(input_dim, config)
        
        self.n_models = config.get('n_models', 5)
        self.model_types = config.get('model_types', ['vae', 'hybrid'])
        self.models = nn.ModuleList()
        
        # Initialize ensemble members
        for _ in range(self.n_models):
            if np.random.choice(['vae', 'hybrid']) == 'vae':
                model = VariationalAutoencoder(input_dim, config)
            else:
                model = HybridImputationModel(input_dim, config)
            self.models.append(model)
        
        # Learned weights for ensemble combination
        self.combination_weights = nn.Parameter(
            torch.ones(self.n_models) / self.n_models
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get predictions from all models
        model_outputs = [model(x, mask) for model in self.models]
        
        # Stack predictions and uncertainties
        predictions = torch.stack([out['imputed'] for out in model_outputs], dim=0)
        uncertainties = torch.stack([out['uncertainty'] for out in model_outputs], dim=0)
        
        # Compute weights based on uncertainties and learned parameters
        weights = F.softmax(self.combination_weights, dim=0)
        weights = weights.view(-1, 1, 1)  # Reshape for broadcasting
        
        # Weighted combination
        ensemble_prediction = (predictions * weights).sum(dim=0)
        
        # Compute ensemble uncertainty (combining aleatory and epistemic uncertainty)
        mean_prediction = ensemble_prediction
        variance = ((predictions - mean_prediction)**2 * weights).sum(dim=0)
        uncertainty = torch.sqrt(variance + (uncertainties * weights).sum(dim=0))
        
        return {
            'imputed': ensemble_prediction,
            'uncertainty': uncertainty,
            'model_predictions': predictions,
            'model_uncertainties': uncertainties,
            'combination_weights': weights.squeeze(),
            'individual_outputs': model_outputs
        }
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Compute individual model losses
        model_losses = []
        for i, model in enumerate(self.models):
            individual_output = outputs['individual_outputs'][i]
            model_loss = model.compute_loss(x, mask, individual_output)
            model_losses.append(model_loss['total_loss'])
        
        # Ensemble loss
        ensemble_mse = F.mse_loss(
            outputs['imputed'][mask],
            x[mask]
        )
        
        # Diversity encouragement loss
        diversity_loss = self._compute_diversity_loss(
            outputs['model_predictions'],
            mask
        )
        
        # Weight regularization
        weight_reg = torch.sum((outputs['combination_weights'] - 1/self.n_models)**2)
        
        # Total loss
        total_loss = (
            ensemble_mse +
            0.1 * torch.stack(model_losses).mean() +
            0.05 * diversity_loss +
            0.01 * weight_reg
        )
        
        return {
            'total_loss': total_loss,
            'mse_loss': ensemble_mse,
            'diversity_loss': diversity_loss,
            'weight_reg': weight_reg,
            'model_losses': torch.stack(model_losses)
        }
    
    def _compute_diversity_loss(
        self,
        predictions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss term encouraging diversity among ensemble members."""
        n_models = predictions.size(0)
        
        # Compute pairwise differences
        diff_matrix = predictions.unsqueeze(0) - predictions.unsqueeze(1)
        diff_matrix = diff_matrix[mask].pow(2)
        
        # Encourage minimum separation between predictions
        min_separation = 0.1
        diversity_loss = torch.relu(min_separation - diff_matrix).mean()
        
        return diversity_loss
    
    def get_model_weights(self) -> torch.Tensor:
        """Get the current weights for ensemble members."""
        return F.softmax(self.combination_weights, dim=0)
    
    def bootstrap_training(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int = 32,
        n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Generate bootstrapped batches for training ensemble members."""
        if n_samples is None:
            n_samples = x.size(0)
            
        # Sample indices with replacement
        indices = torch.randint(0, x.size(0), (n_samples,))
        
        # Create bootstrapped dataset
        x_boot = x[indices]
        mask_boot = mask[indices]
        
        return x_boot, mask_boot

class WeightedEnsemblePredictor:
    """
    Weighted ensemble predictor using model stacking.
    
    This class implements a meta-learning approach to combine
    predictions from multiple models based on their historical performance.
    """
    
    def __init__(
        self,
        base_models: List[BaseModel],
        meta_learning_rate: float = 0.01
    ):
        self.base_models = base_models
        self.n_models = len(base_models)
        self.meta_learning_rate = meta_learning_rate
        
        # Initialize meta-weights
        self.meta_weights = np.ones(self.n_models) / self.n_models
        self.performance_history = []
        
    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate weighted ensemble prediction."""
        predictions = []
        uncertainties = []
        
        # Get predictions from all models
        for model in self.base_models:
            with torch.no_grad():
                output = model(x, mask)
                predictions.append(output['imputed'])
                uncertainties.append(output['uncertainty'])
        
        # Stack predictions and uncertainties
        predictions = torch.stack(predictions, dim=0)
        uncertainties = torch.stack(uncertainties, dim=0)
        
        # Apply meta-weights
        weights = torch.tensor(self.meta_weights, device=x.device)
        weights = weights.view(-1, 1, 1)
        
        # Compute weighted predictions
        weighted_pred = (predictions * weights).sum(dim=0)
        weighted_unc = (uncertainties * weights).sum(dim=0)
        
        return weighted_pred, weighted_unc
    
    def update_weights(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ):
        """Update meta-weights based on model performance."""
        # Compute errors for each model
        errors = [(pred - true_values).pow(2).mean().item() 
                 for pred in predictions]
        
        # Update weights using softmax of negative errors
        neg_errors = -np.array(errors)
        weights = np.exp(neg_errors) / np.sum(np.exp(neg_errors))
        
        # Update using exponential moving average
        self.meta_weights = (
            (1 - self.meta_learning_rate) * self.meta_weights +
            self.meta_learning_rate * weights
        )
        
        # Store performance history
        self.performance_history.append({
            'errors': errors,
            'weights': self.meta_weights.copy()
        })