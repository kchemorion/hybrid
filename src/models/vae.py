import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base import BaseModel
from src.utils.constants import ModelConfig

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
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.shape
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output(attention_output)
        if len(x.shape) == 2:
            output = output.squeeze(1)
        return output, attention_weights


class VAEEncoder(nn.Module):
    """VAE Encoder."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.input_attention = AttentionLayer(input_dim)
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  #LayerNorm instead of BatchNorm1d for linear layers
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                ]
            )
            current_dim = hidden_dim
        self.encoder_layers = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #Added explicit shape check and handling for 2D input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x_attended, attention_weights = self.input_attention(x)

        if len(x_attended.shape) == 3:
            x_attended = x_attended.squeeze(1)
            
        h = self.encoder_layers(x_attended)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        return mu, logvar, attention_weights


class VAEDecoder(nn.Module):
    """VAE Decoder."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        current_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  #LayerNorm instead of BatchNorm1d for linear layers
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                ]
            )
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
    """Variational Autoencoder."""

    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__(input_dim, config)
        self.encoder = VAEEncoder(input_dim, config.hidden_dims, config.latent_dim)
        self.decoder = VAEDecoder(config.latent_dim, config.hidden_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device) #Ensuring eps is on the correct device
        return mu + eps * std

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, attention_weights = self.encoder(x * mask)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decoder(z)
        uncertainty = torch.exp(recon_logvar)
        return {
            "imputed": recon_mu,
            "uncertainty": uncertainty,
            "mu": mu,
            "logvar": logvar,
            "attention_weights": attention_weights,
            "latent": z,
        }

    def compute_loss(
        self, x: torch.Tensor, mask: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        mask = mask.bool()
        recon_loss = F.mse_loss(outputs["imputed"][mask], x[mask]) #Simplified Reconstruction loss
        kl_loss = -0.5 * torch.mean(1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
        total_loss = recon_loss + self.config.kl_weight * kl_loss
        return {"total_loss": total_loss, "reconstruction_loss": recon_loss, "kl_loss": kl_loss}