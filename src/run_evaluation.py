# src/run_evaluation.py

import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import torch
import yaml
import logging
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.hybrid import HybridImputationModel
from src.utils.constants import ModelConfig
from src.utils.visualization import ImputationVisualizer
from src.experiments.evaluator import Evaluator

# Register numpy scalar as safe for loading
add_safe_globals([scalar])

def setup_visualization():
    """Setup visualization settings."""
    # Set default plotting style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.dpi'] = 100

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Setup visualization
        setup_visualization()
        
        # Create output directories
        Path("results/evaluation").mkdir(parents=True, exist_ok=True)
        Path("results/visualizations").mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            logger.info("Config loaded successfully")
        
        # Initialize model
        model_config = ModelConfig(**config['model'])
        model = HybridImputationModel(
            input_dim=config['data']['n_features'],
            config=model_config
        )
        
        # Load checkpoint
        logger.info("Loading checkpoint...")
        checkpoint = torch.load('checkpoints/best_model.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        
        # Create test data
        logger.info("Creating test data...")
        n_samples = 1000
        n_features = config['data']['n_features']
        test_data = torch.randn(n_samples, n_features).to(device)
        test_mask = torch.rand(n_samples, n_features).to(device) > config['data']['missing_ratio']
        
        # Generate predictions
        logger.info("Generating predictions...")
        with torch.no_grad():
            outputs = model(test_data, test_mask)
            imputed_data = outputs['imputed']
            uncertainties = outputs['uncertainty']
        
        # Setup evaluator
        evaluator = Evaluator(
            true_data=test_data,
            mask=test_mask,
            feature_names=[f"Feature_{i}" for i in range(n_features)],
            save_dir="results/evaluation"
        )
        
        # Run evaluation
        logger.info("Running evaluation...")
        evaluator.evaluate_model(
            model_name="hybrid_model",
            imputed_data=imputed_data,
            uncertainties=uncertainties
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = ImputationVisualizer(
            save_dir="results/visualizations",
            style="default",  # Changed from 'seaborn' to 'default'
            context="paper"
        )
        
        visualizer.plot_imputation_results(
            true_values=test_data,
            imputed_values=imputed_data,
            uncertainties=uncertainties,
            feature_names=[f"Feature_{i}" for i in range(n_features)]
        )
        
        # Generate and save report
        logger.info("Generating report...")
        report = evaluator.generate_report()
        report_path = Path("results/evaluation_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation complete. Results saved in results/")
        logger.info(f"- Evaluation report: {report_path}")
        logger.info(f"- Visualizations: results/visualizations/")
        logger.info(f"- Detailed metrics: results/evaluation/")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()