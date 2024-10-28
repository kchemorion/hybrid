import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base import BaseModel
from .vae import VariationalAutoencoder
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
    """Hybrid model combining VAE and Bayesian Network."""

    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__(input_dim, config)
        self.vae = VariationalAutoencoder(input_dim, config)
        self.bn = BayesianNetworkComponent(input_dim)
        self.uncertainty_estimator = UncertaintyEstimator(config.latent_dim, input_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, input_dim), # added +1 for missingness information
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(input_dim, input_dim),
        )
        self.confidence_layer = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.missingness_embedding = nn.Embedding(2, config.latent_dim) #Added missingness embedding layer.

        self.to(self.device)
        self.beta = config.kl_weight
        self.gamma = config.bn_weight
        self.lambda_conf = config.confidence_weight

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        #print(f"x device: {x.device}, mask device: {mask.device}")
        #print(f"VAE device: {next(self.vae.parameters()).device}")
        vae_outputs = self.vae(x * mask, mask)
        
        #print(f"VAE output device: {vae_outputs['imputed'].device}")
        bn_predictions = self.bn.predict(vae_outputs["imputed"].detach().cpu().numpy())
        bn_predictions = torch.tensor(bn_predictions, device=self.device, dtype=torch.float32)

        #print(f"BN predictions device: {bn_predictions.device}")
        # Missingness information
        missingness_info = (1 - mask).long()  # 1 for missing, 0 for observed
        missingness_embedding = self.missingness_embedding(missingness_info)

        # Concatenate for fusion
        combined_predictions = self.fusion_layer(
            torch.cat([vae_outputs["imputed"], bn_predictions, missingness_embedding], dim=-1)
        )
        confidence_weights = self.confidence_layer(combined_predictions)
        final_imputation = confidence_weights * vae_outputs["imputed"] + (1 - confidence_weights) * bn_predictions
        uncertainty = self.uncertainty_estimator(vae_outputs["latent"], vae_outputs["mu"], vae_outputs["logvar"])
        return {
            "imputed": final_imputation,
            "vae_imputed": vae_outputs["imputed"],
            "bn_imputed": bn_predictions,
            "confidence_weights": confidence_weights,
            "uncertainty": uncertainty,
            "attention_weights": vae_outputs["attention_weights"],
            "latent": vae_outputs["latent"],
            "mu": vae_outputs["mu"],
            "logvar": vae_outputs["logvar"],
        }

    def compute_loss(
        self, x: torch.Tensor, mask: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        mask = mask.bool()
        uncertainty = outputs["uncertainty"]
        recon_loss = F.mse_loss(outputs["imputed"][mask], x[mask])  #Simplified reconstruction loss
        kl_loss = -0.5 * torch.mean(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
        bn_loss = F.mse_loss(outputs["bn_imputed"][mask], x[mask])
        conf_loss = F.binary_cross_entropy(outputs["confidence_weights"][mask], mask.float()) #changed confidence loss

        total_loss = recon_loss + self.beta * kl_loss + self.gamma * bn_loss + self.lambda_conf * conf_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "bn_loss": bn_loss,
            "confidence_loss": conf_loss,
        }