# src/models/base.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import numpy as np
from src.utils.constants import ModelConfig  # Add this import

class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all imputation models.
    
    This class defines the interface that all imputation models must implement,
    including basic functionality for training, validation, and prediction.
    """
    def __init__(self, input_dim: int, config: ModelConfig):  # Change type hint
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor with missing values
            mask: Binary mask indicating missing values (1: observed, 0: missing)
            
        Returns:
            Dictionary containing model outputs (imputed values, uncertainties, etc.)
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor, 
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for training.
        
        Args:
            x: Original input tensor
            mask: Missing value mask
            outputs: Model outputs from forward pass
            
        Returns:
            Dictionary containing loss components
        """
        pass
    
    def impute(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor,
        return_uncertainties: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Impute missing values in the input tensor.
        
        Args:
            x: Input tensor with missing values
            mask: Binary mask indicating missing values
            return_uncertainties: Whether to return uncertainty estimates
            
        Returns:
            Tuple of (imputed tensor, uncertainty estimates if requested)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, mask)
            imputed = x * mask + outputs['imputed'] * (1 - mask)
            
            if return_uncertainties:
                return imputed, outputs.get('uncertainty', None)
            return imputed, None
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            mask: Missing value mask
            
        Returns:
            Attention weights if the model uses attention, None otherwise
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, mask)
            return outputs.get('attention_weights', None)
    
    def save_model(self, path: str):
        """Save model parameters and configuration."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'input_dim': self.input_dim
        }, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'BaseModel':
        """Load model from saved state."""
        checkpoint = torch.load(path)
        model = cls(checkpoint['input_dim'], checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model