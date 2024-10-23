# setup.py

from setuptools import setup, find_packages

setup(
    name="hybrid-missing-data-imputation",
    version="0.1.0",
    description="Hybrid Deep Learning Models for Missing Data Imputation in Clinical Trials",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "pytorch-lightning>=1.5.0",
        "numpy>=1.19.2",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "pgmpy>=0.1.14",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "wandb>=0.12.0",
        "pytest>=6.2.5",
        "black>=21.9b0",
        "flake8>=3.9.2",
        "mypy>=0.910",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=6.4.5",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)