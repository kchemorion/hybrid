import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base import BaseModel
from .vae import GMVAE  # Import the GMVAE
from .bayesian import BayesianNetworkComponent
from src.utils.constants import ModelConfig
import numpy as np

class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty in imputed values."""
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, mu, logvar], dim=-1)
        return self.network(combined)

class HybridImputationModel(BaseModel):
    """Hybrid model combining GMVAE and Bayesian Network with GAIN-style training."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__(input_dim, config)
        self.input_dim = input_dim  # Store input_dim
        self.config = config
        
        # GMVAE component
        self.gmv = GMVAE(input_dim + 1, config)
        
        # Bayesian Network component
        self.bn = BayesianNetworkLayer(
            data=data,
            categorical_indices=categorical_indices,
            numerical_indices=numerical_indices
        )
        
        # Generator and Discriminator components
        self.generator = nn.Sequential(
            nn.Linear(input_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, input_dim)
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.uncertainty_estimator = UncertaintyEstimator(config.latent_dim, input_dim)
        self.missingness_embedding = nn.Embedding(2, config.latent_dim)
        
        self.to(self.device)
        self.beta = config.kl_weight
        self.gamma = config.bn_weight

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the GAIN model.
        Args:
            x (torch.Tensor): Input data with missing values (masked).
            mask (torch.Tensor): Mask indicating missing values (1: missing, 0: observed).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Imputed data and hint matrix.
        """
        # Cast mask to float and move to the correct device
        mask = mask.float().to(self.device)
        
        # Generate a hint vector
        random_matrix = torch.rand(x.shape).to(self.device)
        hint_matrix = mask * random_matrix + (1 - mask) * x  # Modified Hint matrix generation
        
        # Combine observed data and hint matrix with appropriate masking
        x_masked = x * (1 - mask)  # Mask observed values for input to generator
        xh = torch.cat([x_masked, hint_matrix], dim=-1)  # Concatenate to form generator input
        
        # Generate initial imputations
        imputed_data = self.generator(xh)
        
        # Get GMVAE outputs
        gmvae_outputs = self.gmv(imputed_data, mask)
        
        # Refine imputations with Bayesian Network
        refined_data = self.bn(imputed_data)
        
        # Final imputation combining all components
        final_imputed = refined_data * mask + x * (1 - mask)
        
        # Calculate uncertainty
        uncertainty = self.uncertainty_estimator(
            gmvae_outputs["latent"],
            gmvae_outputs["mu"],
            gmvae_outputs["logvar"]
        )
        
        # Prepare discriminator input
        imputed_h = torch.cat([final_imputed, hint_matrix], dim=-1)
        
        return {
            "imputed": final_imputed,
            "hint_matrix": hint_matrix,
            "uncertainty": uncertainty,
            "latent": gmvae_outputs["latent"],
            "mu": gmvae_outputs["mu"],
            "logvar": gmvae_outputs["logvar"]
        }

    def discriminate(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Discriminator forward pass."""
        # Combine data and hint vector for discriminator input
        xh = torch.cat([x, h], dim=-1)  # Concatenate actual/imputed data and hint matrix
        # Discriminate
        d_probs = self.discriminator(xh)
        return d_probs

    def compute_loss(
        self, x: torch.Tensor, mask: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute the loss function."""
        # Extract outputs
        imputed_data = outputs["imputed"]
        hint_matrix = outputs["hint_matrix"]
        
        # Discriminator Loss
        d_real = self.discriminate(x, hint_matrix)
        d_fake = self.discriminate(imputed_data.detach(), hint_matrix)  # Detach generator
        d_loss_real = F.binary_cross_entropy_with_logits(d_real, (1 - mask))  # Missing data should be close to 1
        d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, mask)  # Observed data should be close to 0
        d_loss = (d_loss_real + d_loss_fake)
        
        # Generator Loss
        g_loss = F.binary_cross_entropy_with_logits(
            self.discriminate(imputed_data, hint_matrix),
            1 - mask
        )
        
        # Reconstruction loss (only for missing values)
        m = mask.bool().to(self.device)
        recon_loss = F.mse_loss(imputed_data[m], x[m])
        
        # GMVAE losses
        kl_loss = -0.5 * torch.mean(
            1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
        )
        
        # Total generator loss
        total_g_loss = (
            g_loss +
            recon_loss * self.config.alpha +
            kl_loss * self.beta
        )
        
        return {
            "d_loss": d_loss,
            "g_loss": total_g_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss
        }