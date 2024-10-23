# src/utils/visualization.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

class ImputationVisualizer:
    """
    Comprehensive visualization tools for imputation results.
    
    Features:
    - Distribution comparisons
    - Error analysis
    - Uncertainty visualization
    - Feature correlations
    - Missing patterns
    """
    
    def __init__(
        self,
        save_dir: str = "./visualization_results",
        style: str = "seaborn",
        context: str = "paper"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use(style)
        sns.set_context(context)
        
    def plot_imputation_results(
        self,
        true_values: torch.Tensor,
        imputed_values: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None,
        title: str = "Imputation Results"
    ):
        """Create comprehensive visualization of imputation results."""
        self._plot_distribution_comparison(
            true_values, imputed_values, feature_names, title
        )
        self._plot_error_analysis(
            true_values, imputed_values, feature_names
        )
        
        if uncertainties is not None:
            self._plot_uncertainty_analysis(
                true_values, imputed_values, uncertainties, feature_names
            )
            
        self._plot_correlation_analysis(
            true_values, imputed_values, feature_names
        )
        
    def _plot_distribution_comparison(
        self,
        true_values: torch.Tensor,
        imputed_values: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        title: str = "Distribution Comparison"
    ):
        """Plot distribution comparison between true and imputed values."""
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.detach().cpu().numpy()
        if isinstance(imputed_values, torch.Tensor):
            imputed_values = imputed_values.detach().cpu().numpy()
            
        n_features = true_values.shape[1] if len(true_values.shape) > 1 else 1
        feature_names = feature_names or [f"Feature {i+1}" for i in range(n_features)]
        
        fig, axes = plt.subplots(
            n_features, 1,
            figsize=(10, 4 * n_features),
            squeeze=False
        )
        
        for i in range(n_features):
            ax = axes[i, 0]
            
            # Plot distributions
            sns.kdeplot(
                data=true_values[:, i],
                label="True",
                ax=ax
            )
            sns.kdeplot(
                data=imputed_values[:, i],
                label="Imputed",
                ax=ax
            )
            
            ax.set_title(f"{feature_names[i]} Distribution")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        plt.close()
        
    def _plot_error_analysis(
        self,
        true_values: torch.Tensor,
        imputed_values: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ):
        """Plot error analysis visualizations."""
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.detach().cpu().numpy()
        if isinstance(imputed_values, torch.Tensor):
            imputed_values = imputed_values.detach().cpu().numpy()
            
        n_features = true_values.shape[1] if len(true_values.shape) > 1 else 1
        feature_names = feature_names or [f"Feature {i+1}" for i in range(n_features)]
        
        # Compute errors
        errors = true_values - imputed_values
        
        # Error distribution plot
        plt.figure(figsize=(10, 6))
        for i in range(n_features):
            sns.kdeplot(
                data=errors[:, i],
                label=feature_names[i]
            )
        plt.title("Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(self.save_dir / "error_distribution.png")
        plt.close()
        
        # Scatter plot of true vs imputed values
        fig, axes = plt.subplots(
            n_features, 1,
            figsize=(10, 4 * n_features),
            squeeze=False
        )
        
        for i in range(n_features):
            ax = axes[i, 0]
            ax.scatter(true_values[:, i], imputed_values[:, i], alpha=0.5)
            
            # Add perfect prediction line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            
            ax.set_title(f"{feature_names[i]} True vs Imputed")
            ax.set_xlabel("True Values")
# src/utils/visualization.py (continued)

            ax.set_xlabel("True Values")
            ax.set_ylabel("Imputed Values")
            
        plt.tight_layout()
        plt.savefig(self.save_dir / "true_vs_imputed.png")
        plt.close()
        
    def _plot_uncertainty_analysis(
        self,
        true_values: torch.Tensor,
        imputed_values: torch.Tensor,
        uncertainties: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ):
        """Plot uncertainty analysis visualizations."""
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.detach().cpu().numpy()
        if isinstance(imputed_values, torch.Tensor):
            imputed_values = imputed_values.detach().cpu().numpy()
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.detach().cpu().numpy()
            
        n_features = true_values.shape[1] if len(true_values.shape) > 1 else 1
        feature_names = feature_names or [f"Feature {i+1}" for i in range(n_features)]
        
        # Uncertainty vs Error plot
        fig, axes = plt.subplots(
            n_features, 1,
            figsize=(10, 4 * n_features),
            squeeze=False
        )
        
        errors = np.abs(true_values - imputed_values)
        
        for i in range(n_features):
            ax = axes[i, 0]
            
            # Scatter plot with density coloring
            scatter = ax.scatter(
                uncertainties[:, i],
                errors[:, i],
                alpha=0.5,
                c=errors[:, i],
                cmap='viridis'
            )
            
            # Add correlation coefficient
            corr = np.corrcoef(uncertainties[:, i], errors[:, i])[0, 1]
            ax.text(
                0.05, 0.95,
                f'Correlation: {corr:.3f}',
                transform=ax.transAxes,
                verticalalignment='top'
            )
            
            ax.set_title(f"{feature_names[i]} Uncertainty vs Error")
            ax.set_xlabel("Predicted Uncertainty")
            ax.set_ylabel("Absolute Error")
            
            plt.colorbar(scatter, ax=ax)
            
        plt.tight_layout()
        plt.savefig(self.save_dir / "uncertainty_vs_error.png")
        plt.close()
        
        # Calibration plot
        self._plot_calibration_curves(errors, uncertainties, feature_names)
        
    def _plot_calibration_curves(
        self,
        errors: np.ndarray,
        uncertainties: np.ndarray,
        feature_names: List[str],
        n_bins: int = 10
    ):
        """Plot uncertainty calibration curves."""
        n_features = errors.shape[1]
        
        plt.figure(figsize=(10, 6))
        
        for i in range(n_features):
            # Sort by uncertainty
            sort_idx = np.argsort(uncertainties[:, i])
            sorted_errors = errors[sort_idx, i]
            sorted_uncertainties = uncertainties[sort_idx, i]
            
            # Compute bin statistics
            bin_edges = np.linspace(0, len(sorted_errors), n_bins + 1, dtype=int)
            bin_errors = []
            bin_uncertainties = []
            
            for j in range(n_bins):
                start_idx = bin_edges[j]
                end_idx = bin_edges[j + 1]
                
                bin_errors.append(np.mean(sorted_errors[start_idx:end_idx]))
                bin_uncertainties.append(np.mean(sorted_uncertainties[start_idx:end_idx]))
                
            plt.plot(
                bin_uncertainties,
                bin_errors,
                'o-',
                label=feature_names[i]
            )
            
        # Add diagonal line for perfect calibration
        lims = [
            np.min([plt.xlim()[0], plt.ylim()[0]]),
            np.max([plt.xlim()[1], plt.ylim()[1]])
        ]
        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal')
        
        plt.title("Uncertainty Calibration")
        plt.xlabel("Predicted Uncertainty")
        plt.ylabel("Empirical Error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / "calibration_curves.png")
        plt.close()
        
    def _plot_correlation_analysis(
        self,
        true_values: torch.Tensor,
        imputed_values: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ):
        """Plot correlation analysis visualizations."""
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.detach().cpu().numpy()
        if isinstance(imputed_values, torch.Tensor):
            imputed_values = imputed_values.detach().cpu().numpy()
            
        n_features = true_values.shape[1] if len(true_values.shape) > 1 else 1
        feature_names = feature_names or [f"Feature {i+1}" for i in range(n_features)]
        
        # Compute correlation matrices
        true_corr = np.corrcoef(true_values.T)
        imputed_corr = np.corrcoef(imputed_values.T)
        
        # Plot correlation matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(
            true_corr,
            ax=ax1,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            xticklabels=feature_names,
            yticklabels=feature_names
        )
        ax1.set_title("True Data Correlations")
        
        sns.heatmap(
            imputed_corr,
            ax=ax2,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            xticklabels=feature_names,
            yticklabels=feature_names
        )
        ax2.set_title("Imputed Data Correlations")
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "correlation_analysis.png")
        plt.close()
        
        # Plot correlation difference
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            imputed_corr - true_corr,
            cmap='RdBu_r',
            vmin=-0.5,
            vmax=0.5,
            xticklabels=feature_names,
            yticklabels=feature_names
        )
        plt.title("Correlation Difference (Imputed - True)")
        plt.tight_layout()
        plt.savefig(self.save_dir / "correlation_difference.png")
        plt.close()
        
    def plot_missing_patterns(
        self,
        mask: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ):
        """Plot missing data patterns."""
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        n_features = mask.shape[1] if len(mask.shape) > 1 else 1
        feature_names = feature_names or [f"Feature {i+1}" for i in range(n_features)]
        
        # Missing pattern heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            mask.T,
            cmap='binary',
            yticklabels=feature_names,
            cbar_kws={'label': 'Missing'}
        )
        plt.title("Missing Data Patterns")
        plt.xlabel("Sample Index")
        plt.tight_layout()
        plt.savefig(self.save_dir / "missing_patterns.png")
        plt.close()
        
        # Missing rate by feature
        plt.figure(figsize=(10, 4))
        missing_rates = 1 - mask.mean(axis=0)
        
        sns.barplot(
            x=list(range(n_features)),
            y=missing_rates,
            palette='viridis'
        )
        plt.xticks(range(n_features), feature_names, rotation=45)
        plt.title("Missing Rate by Feature")
        plt.xlabel("Feature")
        plt.ylabel("Missing Rate")
        plt.tight_layout()
        plt.savefig(self.save_dir / "missing_rates.png")
        plt.close()
        
    def create_summary_dashboard(
        self,
        results_dict: Dict[str, Dict],
        save_name: str = "summary_dashboard.png"
    ):
        """Create a comprehensive dashboard of imputation results."""
        n_models = len(results_dict)
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, n_models)
        
        # Plot metrics comparison
        ax_metrics = fig.add_subplot(gs[0, :])
        metrics_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in results_dict.items()
        }).T
        
        metrics_df.plot(kind='bar', ax=ax_metrics)
        ax_metrics.set_title("Model Comparison - Metrics")
        ax_metrics.set_xticklabels(ax_metrics.get_xticklabels(), rotation=45)
        
        # Plot error distributions
        for i, (model_name, result) in enumerate(results_dict.items()):
            ax = fig.add_subplot(gs[1, i])
            errors = result.get('errors', None)
            
            if errors is not None:
                sns.kdeplot(data=errors.flatten(), ax=ax)
                ax.set_title(f"{model_name}\nError Distribution")
                
        # Plot uncertainty calibration if available
        for i, (model_name, result) in enumerate(results_dict.items()):
            ax = fig.add_subplot(gs[2, i])
            
            if 'calibration' in result:
                predicted = result['calibration']['predicted']
                empirical = result['calibration']['empirical']
                
                ax.scatter(predicted, empirical, alpha=0.5)
                ax.plot([0, 1], [0, 1], 'r--', alpha=0.75)
                ax.set_title(f"{model_name}\nUncertainty Calibration")
                
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()