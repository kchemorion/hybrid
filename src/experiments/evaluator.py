# src/experiments/evaluator.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Evaluator:
    """
    Comprehensive evaluation framework for imputation models.
    
    Features:
    - Multiple evaluation metrics
    - Uncertainty analysis
    - Visualization tools
    - Comparison across models
    - Statistical tests
    """
    
    def __init__(
        self,
        true_data: torch.Tensor,
        mask: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        save_dir: str = "./evaluation_results"
    ):
        self.true_data = true_data
        self.mask = mask
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(true_data.shape[1])]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
        
    def evaluate_model(
        self,
        model_name: str,
        imputed_data: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ):
        """
        Evaluate imputation results for a single model.
        
        Args:
            model_name: Name of the model
            imputed_data: Imputed values
            uncertainties: Uncertainty estimates (optional)
        """
        # Compute basic metrics
        metrics = self._compute_basic_metrics(imputed_data)
        
        # Compute feature-wise metrics
        feature_metrics = self._compute_feature_metrics(imputed_data)
        
        # Analyze uncertainties if provided
        if uncertainties is not None:
            uncertainty_metrics = self._analyze_uncertainties(imputed_data, uncertainties)
            metrics.update(uncertainty_metrics)
            
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'feature_metrics': feature_metrics,
            'imputed_data': imputed_data.cpu().numpy(),
            'uncertainties': uncertainties.cpu().numpy() if uncertainties is not None else None
        }
        
    def compare_models(self, significance_level: float = 0.05) -> pd.DataFrame:
        """Compare performance across different models."""
        if len(self.results) < 2:
            raise ValueError("Need at least two models to compare")
            
        comparison_df = pd.DataFrame()
        
        # Collect metrics for all models
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_df[model_name] = pd.Series(metrics)
            
        # Perform statistical tests
        if len(self.results) > 2:
            # Perform Friedman test
            from scipy.stats import friedmanchisquare
            
            metric_pvalues = {}
            for metric in comparison_df.index:
                values = [result['metrics'][metric] for result in self.results.values()]
                stat, pval = friedmanchisquare(*values)
                metric_pvalues[metric] = pval
                
            comparison_df['friedman_pvalue'] = pd.Series(metric_pvalues)
            
        # Generate comparison plots
        self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def _compute_basic_metrics(self, imputed_data: np.ndarray) -> Dict[str, float]:
        """Compute basic imputation metrics."""
        missing_mask = self.mask.astype(bool) #corrected to bool
        true_missing = self.true_data[missing_mask]
        imputed_missing = imputed_data[missing_mask]
        rmse = np.sqrt(mean_squared_error(true_missing, imputed_missing))
        mae = mean_absolute_error(true_missing, imputed_missing)
        try:
          correlation = pearsonr(true_missing.flatten(), imputed_missing.flatten())[0]
        except ValueError:
          correlation = np.nan #Handle cases where Pearson correlation is undefined.
        r2 = r2_score(true_missing, imputed_missing) #Added R-squared
        return {"rmse": rmse, "mae": mae, "correlation": correlation, "r2":r2}

    def _compute_feature_metrics(self, imputed_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute feature-wise metrics."""
        feature_metrics = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_mask = self.mask[:, i].astype(bool) #corrected to bool
            true_values = self.true_data[feature_mask, i]
            imputed_values = imputed_data[feature_mask, i]
            rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
            mae = mean_absolute_error(true_values, imputed_values)
            try:
              correlation = pearsonr(true_values, imputed_values)[0]
            except ValueError:
              correlation = np.nan
            missing_ratio = 1 - np.mean(feature_mask)
            feature_metrics[feature_name] = {
                "rmse": rmse,
                "mae": mae,
                "correlation": correlation,
                "missing_ratio": missing_ratio,
            }
        return feature_metrics
        
    def _analyze_uncertainties(
        self,
        imputed_data: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Dict[str, float]:
        """Analyze uncertainty estimates."""
        missing_mask = ~self.mask
        true_missing = self.true_data[missing_mask]
        imputed_missing = imputed_data[missing_mask]
        uncertainties_missing = uncertainties[missing_mask]
        
        # Compute error-uncertainty correlation
        errors = torch.abs(true_missing - imputed_missing)
        error_uncertainty_corr = pearsonr(
            errors.cpu().numpy(),
            uncertainties_missing.cpu().numpy()
        )[0]
        
        # Compute calibration metrics
        calibration_metrics = self._compute_uncertainty_calibration(
            true_missing,
            imputed_missing,
            uncertainties_missing
        )
        
        return {
            'error_uncertainty_correlation': error_uncertainty_corr,
            **calibration_metrics
        }
        
    def _compute_uncertainty_calibration(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        # Sort by uncertainty
        sorted_indices = torch.argsort(uncertainties)
        sorted_true = true_values[sorted_indices]
        sorted_pred = predictions[sorted_indices]
        sorted_unc = uncertainties[sorted_indices]
        
        # Divide into bins
        bin_size = len(sorted_unc) // n_bins
        calibration_error = 0.0
        sharpness = uncertainties.mean().item()
        
        bin_metrics = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_unc)
            
            # Compute empirical error in bin
            bin_errors = torch.abs(
                sorted_true[start_idx:end_idx] - 
                sorted_pred[start_idx:end_idx]
            )
            empirical_error = bin_errors.mean().item()
            
            # Get mean uncertainty in bin
            predicted_error = sorted_unc[start_idx:end_idx].mean().item()
            
            # Update calibration error
            calibration_error += abs(empirical_error - predicted_error)
            
            bin_metrics.append({
                'bin_id': i,
                'empirical_error': empirical_error,
                'predicted_error': predicted_error,
                'size': end_idx - start_idx
            })
            
        calibration_error /= n_bins
        
        return {
            'calibration_error': calibration_error,
            'sharpness': sharpness,
            'bin_metrics': bin_metrics
        }
        
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Create comparison plots for different models."""
        # Bar plot for main metrics
        plt.figure(figsize=(12, 6))
        comparison_df.drop('friedman_pvalue', axis=1, errors='ignore').plot(kind='bar')
        plt.title('Model Comparison - Main Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_comparison_metrics.png')
        plt.close()
        
        # Feature-wise comparison
        self._plot_feature_comparison()
        
        # Uncertainty analysis if available
        self._plot_uncertainty_comparison()
        
    def _plot_feature_comparison(self):
        """Create feature-wise comparison plots."""
        n_models = len(self.results)
        n_features = len(self.feature_names)
        
        # RMSE comparison
        plt.figure(figsize=(12, 6))
        feature_rmse = pd.DataFrame(
            {model_name: [result['feature_metrics'][feat]['rmse'] 
                         for feat in self.feature_names]
             for model_name, result in self.results.items()},
            index=self.feature_names
        )
        
        sns.heatmap(feature_rmse, annot=True, cmap='YlOrRd')
        plt.title('Feature-wise RMSE Comparison')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'feature_rmse_comparison.png')
        plt.close()
        
    def _plot_uncertainty_comparison(self):
        """Create uncertainty analysis plots."""
        models_with_uncertainty = {
            name: result for name, result in self.results.items()
            if result['uncertainties'] is not None
        }
        
        if not models_with_uncertainty:
            return
            
        # Calibration plots
        plt.figure(figsize=(12, 6))
        for model_name, result in models_with_uncertainty.items():
            bin_metrics = result['metrics']['bin_metrics']
            empirical_errors = [m['empirical_error'] for m in bin_metrics]
            predicted_errors = [m['predicted_error'] for m in bin_metrics]
            
            plt.plot(
                predicted_errors,
                empirical_errors,
                'o-',
                label=f"{model_name} (CE={result['metrics']['calibration_error']:.3f})"
            )
            
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Predicted Error')
        plt.ylabel('Empirical Error')
        plt.title('Uncertainty Calibration')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.save_dir / 'uncertainty_calibration.png')
        plt.close()
        
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        report = ["# Imputation Model Evaluation Report\n"]
        
        # Single model evaluation
        if len(self.results) == 1:
            model_name = list(self.results.keys())[0]
            result = self.results[model_name]
            
            # Overall metrics
            report.append("## Model Metrics\n")
            metrics_df = pd.DataFrame({
                'Metric': result['metrics'].keys(),
                'Value': result['metrics'].values()
            })
            report.append(metrics_df.to_markdown(index=False))
            report.append("\n")
            
            # Feature-wise analysis
            report.append("## Feature-wise Analysis\n")
            feature_df = pd.DataFrame.from_dict(
                result['feature_metrics'],
                orient='index'
            )
            report.append(feature_df.to_markdown())
            report.append("\n")
            
            # Uncertainty analysis if available
            if result.get('uncertainties') is not None:
                report.append("## Uncertainty Analysis\n")
                uncertainty_metrics = {
                    k: v for k, v in result['metrics'].items()
                    if k in ['calibration_error', 'sharpness', 'error_uncertainty_correlation']
                }
                uncertainty_df = pd.DataFrame({
                    'Metric': uncertainty_metrics.keys(),
                    'Value': uncertainty_metrics.values()
                })
                report.append(uncertainty_df.to_markdown(index=False))
                report.append("\n")
        
        # Multiple model comparison
        else:
            report.append("## Model Comparison\n")
            try:
                comparison_df = self.compare_models()
                report.append(comparison_df.to_markdown())
                report.append("\n")
            except Exception as e:
                report.append(f"Error generating model comparison: {str(e)}\n")
        
        return '\n'.join(report)
