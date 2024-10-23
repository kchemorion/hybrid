
# Deep Hybrid Models for Missing Data Imputation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A sophisticated framework for missing data imputation using hybrid deep learning models. This project combines Variational Autoencoders (VAEs) with Bayesian Networks to create a robust and accurate imputation system, particularly suited for clinical trial data.

## Features

- **Hybrid Architecture**
  - Variational Autoencoders (VAEs) for complex pattern learning
  - Bayesian Networks for uncertainty modeling
  - Attention mechanisms for feature relationships
  - Ensemble methods for robust predictions

- **Multiple Missing Data Mechanisms**
  - Missing Completely at Random (MCAR)
  - Missing at Random (MAR)
  - Missing Not at Random (MNAR)
  - Custom missing patterns

- **Comprehensive Evaluation**
  - Multiple evaluation metrics
  - Uncertainty quantification
  - Visualization tools
  - Statistical analysis

- **Advanced Features**
  - Uncertainty-aware imputation
  - Feature importance analysis
  - Distribution preservation
  - Correlation maintenance

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)

### Setting Up the Environment

# Clone the repository
git clone https://github.com/kchemorion/missing-data-imputation.git
cd missing-data-imputation

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt


### Optional: Docker Support


# Build the Docker image
docker build -t missing-data-imputation .

# Run the container
docker run -it --gpus all missing-data-imputation


## Project Structure

```
missing_data_imputation/
├── src/
│   ├── models/          # Model implementations
│   ├── data/           # Data handling utilities
│   ├── experiments/    # Training and evaluation
│   └── utils/          # Helper functions
├── configs/            # Configuration files
├── notebooks/         # Jupyter notebooks
├── tests/            # Unit tests
└── README.md
```

## Quick Start

1. **Configure Your Experiment**

Edit `configs/config.yaml` to set up your experiment:

```yaml
model:
  model_type: "hybrid"
  hidden_dims: [256, 128, 64]
  latent_dim: 32
  # ... other parameters

experiment:
  experiment_name: "my_experiment"
  # ... other settings
```

2. **Run Training**

```bash
python src/train.py --config configs/config.yaml
```

3. **View Results**

```bash
python src/analyze_results.py --experiment_name "my_experiment"
```

## Usage Examples

### Basic Usage

```python
from src.models import HybridImputationModel
from src.data import MissingValueDataset

# Load and prepare data
dataset = MissingValueDataset(data, mask)

# Create and train model
model = HybridImputationModel(input_dim=data.shape[1])
trainer = Trainer(model, dataset)
trainer.train()

# Impute missing values
imputed_data = model.impute(data, mask)
```

### Advanced Usage

```python
# Custom missing pattern
dataset = MissingValueDataset(
    data,
    missing_mechanism='custom',
    pattern_generator=your_pattern_function
)

# Ensemble model with uncertainty
model = EnsembleModel(
    base_models=['vae', 'hybrid'],
    n_models=5
)

# Train with advanced features
trainer = Trainer(
    model,
    dataset,
    uncertainty_aware=True,
    feature_importance=True
)

# Get imputed values with uncertainty estimates
imputed_data, uncertainties = model.impute(
    data,
    mask,
    return_uncertainties=True
)
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Model Architecture](docs/models.md)
- [Data Preprocessing](docs/data.md)
- [Training Process](docs/training.md)
- [Evaluation Metrics](docs/evaluation.md)
- [API Reference](docs/api.md)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

## Citing This Work

If you use this code in your research, please cite:

```bibtex
@software{hybrid_imputation_2024,
  title = {Deep Hybrid Models for Missing Data Imputation},
  author = {Francis Kiptengwer Chemorion},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/kchemorion/missing-data-imputation}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors
- Special thanks to the PyTorch and pgmpy teams
- Inspiration from various research papers (see [Citations](docs/citations.md))

## Support

For support, please:
1. Check the [Documentation](docs/)
2. Look through [Existing Issues](https://github.com/kchemorion/missing-data-imputation/issues)
3. Open a [New Issue](https://github.com/kchemorion/missing-data-imputation/issues/new)

## Roadmap

- [ ] Add support for temporal data
- [ ] Implement more sophisticated attention mechanisms
- [ ] Add distributed training support
- [ ] Improve uncertainty calibration
- [ ] Add more visualization tools

## Contact

- Your Name - [kchemorion@gmail.com](mailto:kchemorion@gmail.com)
- Project Link: [https://github.com/kchemorion/missing-data-imputation](https://github.com/kchemorion/missing-data-imputation)
    

