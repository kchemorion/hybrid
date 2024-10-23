# src/models/hybrid.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base import BaseModel
from .vae import VariationalAutoencoder
from .bayesian import BayesianNetworkComponent
from src.utils.constants import ModelConfig  # Add this import
import numpy as np

class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty in imputed values."""
    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),  # Added another layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 20),  # Changed to match feature dimension
            nn.Softplus()
        )
        
    def forward(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, mu, logvar], dim=-1)
        return self.network(combined)

class HybridImputationModel(BaseModel):
    """
    Hybrid model combining VAE and Bayesian Network for missing data imputation.
    
    This model integrates:
    1. VAE for learning complex data distributions
    2. Bayesian Network for capturing variable dependencies
    3. Uncertainty estimation for imputed values
    4. Attention mechanism for feature relationships
    """
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__(input_dim, config)
        
        # Initialize components
        self.vae = VariationalAutoencoder(input_dim, config)
        self.bn = BayesianNetworkComponent(input_dim)
        self.uncertainty_estimator = UncertaintyEstimator(config.latent_dim)
        
        # Correct the fusion layer dimensions
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # First layer accepts concatenated features
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_dim, input_dim)  # Output same dimension as input
        )
        
        # Confidence weighting layer
        self.confidence_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Move device setting here
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)
        
        # Training parameters
        self.beta = 1.0  # VAE KL weight
        self.gamma = 0.5  # BN weight
        self.lambda_conf = 0.1  # Confidence loss weight

    def _convert_beliefs_to_predictions(self, bn_beliefs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert Bayesian network beliefs to continuous predictions."""
        try:
            # Convert list to numpy array first for efficiency
            beliefs_list = [beliefs.numpy() if isinstance(beliefs, torch.Tensor) else beliefs 
                        for beliefs in bn_beliefs.values()]
            beliefs_array = np.stack(beliefs_list)
            
            # Convert to tensor
            beliefs_tensor = torch.tensor(
                beliefs_array,
                device=self.device,
                dtype=torch.float32
            )
            
            # Get batch size from original input
            batch_size = self.last_batch_size  # We'll set this in forward()
            
            # Create value points
            value_points = torch.linspace(0, 1, beliefs_tensor.size(-1), device=self.device)
            
            # Compute weighted average for each feature
            continuous_predictions = (beliefs_tensor * value_points).sum(-1)
            
            # Reshape to match batch size [batch_size, n_features]
            continuous_predictions = continuous_predictions.unsqueeze(0).expand(batch_size, -1)
            
            return continuous_predictions
            
        except Exception as e:
            print(f"Shape debug - beliefs_array: {beliefs_array.shape}")
            print(f"Shape debug - beliefs_tensor: {beliefs_tensor.shape}")
            print(f"Shape debug - continuous_predictions: {continuous_predictions.shape}")
            raise e

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        print(f"x device: {x.device}, mask device: {mask.device}")
        print(f"VAE device: {next(self.vae.parameters()).device}")
        # print(f"BayesianNetwork device: {next(self.bn.parameters()).device}")
        
        vae_outputs = self.vae(x, mask)
        print(f"VAE output device: {vae_outputs['imputed'].device}")
        
        bn_predictions = self.bn.predict(vae_outputs['imputed'])
        print(f"BN predictions device: {bn_predictions.device}")
        
        # Store batch size for belief conversion
        self.last_batch_size = x.size(0)
                
        # Update Bayesian network beliefs and get predictions
        bn_beliefs = self.bn.update_beliefs(vae_outputs['latent'], x * mask)
        bn_predictions = self._convert_beliefs_to_predictions(bn_beliefs)
        
        # Print shapes for debugging
        print(f"vae_outputs['imputed'] shape: {vae_outputs['imputed'].shape}")
        print(f"bn_predictions shape: {bn_predictions.shape}")
        
        # Ensure shapes match for concatenation
        if bn_predictions.size(0) != vae_outputs['imputed'].size(0):
            bn_predictions = bn_predictions.expand(vae_outputs['imputed'].size(0), -1)
        
        # Combine VAE and BN predictions
        combined_predictions = self.fusion_layer(
            torch.cat([vae_outputs['imputed'], bn_predictions], dim=-1)
        )
        
        # Rest of forward method...
        confidence_weights = self.confidence_layer(combined_predictions)
        
        final_imputation = (
            confidence_weights * vae_outputs['imputed'] +
            (1 - confidence_weights) * bn_predictions
        )
        
        uncertainty = self.uncertainty_estimator(
            vae_outputs['latent'],
            vae_outputs['mu'],
            vae_outputs['logvar']
        )
        
        return {
            'imputed': final_imputation,
            'vae_imputed': vae_outputs['imputed'],
            'bn_imputed': bn_predictions,
            'confidence_weights': confidence_weights,
            'uncertainty': uncertainty,
            'attention_weights': vae_outputs['attention_weights'],
            'latent': vae_outputs['latent'],
            'mu': vae_outputs['mu'],
            'logvar': vae_outputs['logvar']
        }
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss for the hybrid model."""
        # Convert mask to boolean tensor
        mask = mask.bool()
        
        # Ensure uncertainty has correct shape
        uncertainty = outputs['uncertainty']
        
        # VAE reconstruction loss - mask out missing values
        recon_loss = F.gaussian_nll_loss(
            outputs['imputed'][mask],
            x[mask],
            torch.clamp(uncertainty[mask], min=1e-6)
        )
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(  # Use mean instead of sum
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        )
        
        # Bayesian network consistency loss - mask out missing values
        bn_loss = F.mse_loss(
            outputs['bn_imputed'][mask],
            x[mask]
        )
        
        # Confidence regularization loss
        conf_loss = torch.mean(
            -outputs['confidence_weights'] * torch.log(outputs['confidence_weights'] + 1e-10)
            -(1 - outputs['confidence_weights']) * torch.log(1 - outputs['confidence_weights'] + 1e-10)
        )
        
        # Uncertainty calibration loss - only for observed values
        uncertainty_loss = self._compute_uncertainty_calibration_loss(
            x[mask],
            outputs['imputed'][mask],
            uncertainty[mask]
        )
        
        # Ensure all losses are tensors that require grad
        losses = {
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'bn_loss': bn_loss,
            'confidence_loss': conf_loss,
            'uncertainty_loss': uncertainty_loss
        }
        
        # Compute total loss as weighted sum
        total_loss = (
            recon_loss +
            self.beta * kl_loss +
            self.gamma * bn_loss +
            self.lambda_conf * conf_loss +
            0.1 * uncertainty_loss
        )
        
        # Add total loss to dictionary
        losses['total_loss'] = total_loss
        
        return losses

    def _compute_uncertainty_calibration_loss(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss to encourage well-calibrated uncertainty estimates."""
        squared_errors = (true_values - predictions).pow(2)
        uncertainty_loss = torch.mean(
            (squared_errors - uncertainties.clamp(min=1e-6)).pow(2)
        )
        return uncertainty_loss
    
    def get_importance_weights(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get feature importance weights from attention mechanism."""
        with torch.no_grad():
            outputs = self.forward(x, mask)
            return outputs['attention_weights'].mean(dim=0)
    
    def get_imputation_confidence(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get confidence scores and uncertainty estimates for imputed values.
        
        Returns:
            Tuple of (confidence_scores, uncertainty_estimates)
        """
        with torch.no_grad():
            outputs = self.forward(x, mask)
            return outputs['confidence_weights'], outputs['uncertainty']

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vae = VAE(config)
        self.bayesian_network = BayesianNetwork(config)
        
    def to(self, device):
        super().to(device)
        self.vae.to(device)
        self.bayesian_network.to(device)
        return self

    def forward(self, x, mask):
        # ... existing code ...
        vae_outputs = self.vae(x, mask)
        
        # Replace this line:
        # bn_predictions = self.bn(vae_outputs['imputed'])
        
        # With something like this:
        bn_predictions = self.bn.predict(vae_outputs['imputed'])
        
        # ... rest of the method ...
