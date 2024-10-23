# src/models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base import BaseModel
from src.utils.constants import ModelConfig  # Add this import

class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads
        assert self.head_dim * n_heads == input_dim, "input_dim must be divisible by n_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure input is 3D [batch_size, seq_len, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output(attention_output)
        
        # If input was 2D, return 2D output
        if len(x.shape) == 2:
            output = output.squeeze(1)
            
        return output, attention_weights

# src/models/vae.py

class VAEEncoder(nn.Module):
    """Encoder network with attention mechanism."""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        self.input_attention = AttentionLayer(input_dim)
        
        # Build encoder layers
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # This is where the error occurs
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        self.encoder_layers = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Handle the case where input is 2D [batch_size, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, features]
        
        # Apply attention
        x_attended, attention_weights = self.input_attention(x)
        
        # Reshape for batch norm (expects [batch_size, features])
        if len(x_attended.shape) == 3:
            x_attended = x_attended.squeeze(1)
            
        # Encode
        h = self.encoder_layers(x_attended)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar, attention_weights

class VAEDecoder(nn.Module):
    """Probabilistic decoder network."""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        
        # Build decoder layers
        layers = []
        current_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        self.decoder_layers = nn.Sequential(*layers)
        self.output_mu = nn.Linear(hidden_dims[0], output_dim)
        self.output_logvar = nn.Linear(hidden_dims[0], output_dim)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder_layers(z)
        mu = self.output_mu(h)
        logvar = self.output_logvar(h)
        return mu, logvar

class VariationalAutoencoder(BaseModel):
    """Complete VAE model for missing data imputation."""
    def __init__(self, input_dim: int, config: ModelConfig):  # Change type hint
        super().__init__(input_dim, config)
        
        # Access attributes directly from ModelConfig
        hidden_dims = config.hidden_dims
        latent_dim = config.latent_dim
        
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, input_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        mu, logvar, attention_weights = self.encoder(x * mask)
        
        # Sample latent representation
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_mu, recon_logvar = self.decoder(z)
        
        # Compute imputation uncertainty
        uncertainty = torch.exp(recon_logvar)
        
        return {
            'imputed': recon_mu,
            'uncertainty': uncertainty,
            'mu': mu,
            'logvar': logvar,
            'attention_weights': attention_weights,
            'latent': z
        }
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss components."""
        # Reconstruction loss (only for observed values)
        recon_loss = F.gaussian_nll_loss(
            outputs['imputed'][mask], 
            x[mask],
            outputs['uncertainty'][mask]
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        )
        
        # Total loss
        total_loss = recon_loss + self.config.get('beta', 1.0) * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }