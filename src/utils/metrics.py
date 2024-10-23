# src/utils/metrics.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import warnings

class ImputationMetrics:
    """
    Comprehensive metrics for evaluating imputation quality.
    
    Features:
    - Basic error metrics (RMSE, MAE)
    - Correlation metrics
    - Distribution metrics
    - Uncertainty metrics
    - Feature-wise evaluation
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def compute_metrics(
        self,
        true_values: torch.Tensor,
        imputed_values: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        feature_wise: bool = False
    ) -> Dict[str, float]:
        """
        Compute comprehensive imputation metrics.
        
        Args:
            true_values: Ground truth values
            imputed_values: Imputed values
            uncertainties: Uncertainty estimates (optional)
            feature_wise: Whether to compute metrics per feature
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy if needed
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.detach().cpu().numpy()
        if isinstance(imputed_values, torch.Tensor):
            imputed_values = imputed_values.detach().cpu().numpy()
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.detach().cpu().numpy()
            
        # Basic error metrics
        metrics.update(self._compute_error_metrics(true_values, imputed_values))
        
        # Correlation metrics
        metrics.update(self._compute_correlation_metrics(true_values, imputed_values))
        
        # Distribution metrics
        metrics.update(self._compute_distribution_metrics(true_values, imputed_values))
        
        # Uncertainty metrics if provided
        if uncertainties is not None:
            metrics.update(self._compute_uncertainty_metrics(
                true_values, imputed_values, uncertainties
            ))
            
        # Feature-wise metrics if requested
        if feature_wise and len(true_values.shape) > 1:
            feature_metrics = self._compute_feature_wise_metrics(
                true_values, imputed_values, uncertainties
            )
            metrics['feature_metrics'] = feature_metrics
            
        self.metrics_history.append(metrics)
        return metrics
    
    def _compute_error_metrics(
        self,
        true_values: np.ndarray,
        imputed_values: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic error metrics."""
        # Handle potential warnings for empty arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            mse = np.mean((true_values - imputed_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(true_values - imputed_values))
            mape = np.mean(np.abs((true_values - imputed_values) / (true_values + 1e-7))) * 100
            
            # Normalized metrics
            range_values = np.max(true_values) - np.min(true_values)
            nrmse = rmse / (range_values + 1e-7)
            
            # R-squared
            r2 = r2_score(true_values, imputed_values)
            
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'nrmse': nrmse,
            'r2': r2
        }
        
    def _compute_correlation_metrics(
        self,
        true_values: np.ndarray,
        imputed_values: np.ndarray
    ) -> Dict[str, float]:
        """Compute correlation-based metrics."""
        # Flatten arrays if needed
        if len(true_values.shape) > 1:
            true_values = true_values.flatten()
            imputed_values = imputed_values.flatten()
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Pearson correlation
            pearson_corr, _ = pearsonr(true_values, imputed_values)
            
            # Spearman correlation
            spearman_corr, _ = spearmanr(true_values, imputed_values)
            
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        }
        
    def _compute_distribution_metrics(
        self,
        true_values: np.ndarray,
        imputed_values: np.ndarray
    ) -> Dict[str, float]:
        """Compute distribution-based metrics."""
        # KL divergence between histograms
        hist_true, bins = np.histogram(true_values, bins=50, density=True)
        hist_imputed, _ = np.histogram(imputed_values, bins=bins, density=True)
        
        # Add small constant to avoid division by zero
        eps = 1e-10
        hist_true = hist_true + eps
        hist_imputed = hist_imputed + eps
        
        kl_div = np.sum(hist_true * np.log(hist_true / hist_imputed))
        
        # Distribution statistics comparison
        true_mean = np.mean(true_values)
        true_std = np.std(true_values)
        imputed_mean = np.mean(imputed_values)
        imputed_std = np.std(imputed_values)
        
        mean_diff = np.abs(true_mean - imputed_mean)
        std_diff = np.abs(true_std - imputed_std)
        
        return {
            'kl_divergence': kl_div,
            'mean_difference': mean_diff,
            'std_difference': std_diff
        }
        
    def _compute_uncertainty_metrics(
        self,
        true_values: np.ndarray,
        imputed_values: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """Compute uncertainty-related metrics."""
        # Absolute errors
        errors = np.abs(true_values - imputed_values)
        
        # Error-uncertainty correlation
        error_unc_corr, _ = pearsonr(errors.flatten(), uncertainties.flatten())
        
        # Calibration metrics
        calibration_metrics = self._compute_calibration_metrics(
            errors.flatten(),
            uncertainties.flatten()
        )
        
        metrics = {
            'error_uncertainty_correlation': error_unc_corr,
            'uncertainty_mean': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties)
        }
        metrics.update(calibration_metrics)
        
        return metrics
        
    def _compute_calibration_metrics(
        self,
        errors: np.ndarray,
        uncertainties: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        # Sort by uncertainty
        sort_idx = np.argsort(uncertainties)
        sorted_errors = errors[sort_idx]
        sorted_uncertainties = uncertainties[sort_idx]
        
        # Divide into bins
        bin_size = len(sorted_errors) // n_bins
        calibration_error = 0.0
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(sorted_errors)
            
            bin_errors = sorted_errors[start_idx:end_idx]
            bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
            
            empirical_error = np.mean(bin_errors)
            predicted_error = np.mean(bin_uncertainties)
            
            calibration_error += np.abs(empirical_error - predicted_error)
            
        calibration_error /= n_bins
        
        return {
            'calibration_error': calibration_error
        }
        
    def _compute_feature_wise_metrics(
        self,
        true_values: np.ndarray,
        imputed_values: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each feature separately."""
        n_features = true_values.shape[1]
        feature_metrics = {}
        
        for i in range(n_features):
            feature_true = true_values[:, i]
            feature_imputed = imputed_values[:, i]
            feature_uncertainties = uncertainties[:, i] if uncertainties is not None else None
            
            metrics = {}
            # Error metrics
            metrics.update(self._compute_error_metrics(feature_true, feature_imputed))
            
            # Correlation metrics
            metrics.update(self._compute_correlation_metrics(feature_true, feature_imputed))
            
            # Distribution metrics
            metrics.update(self._compute_distribution_metrics(feature_true, feature_imputed))
            
            # Uncertainty metrics if provided
            if feature_uncertainties is not None:
                metrics.update(self._compute_uncertainty_metrics(
                    feature_true, feature_imputed, feature_uncertainties
                ))
                
            feature_metrics[f'feature_{i}'] = metrics
            
        return feature_metrics
        
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics over metrics history."""
        if not self.metrics_history:
            return {}
            
        summary = {}
        metrics = self.metrics_history[0].keys()
        
        for metric in metrics:
            if metric != 'feature_metrics':
                values = [h[metric] for h in self.metrics_history]
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
                
        return summary