# 🏭 Production ML Models

> The models that actually make money in industry — anomaly detection, gradient boosting, and Gaussian processes.

## Overview

This project implements **5 production-grade ML models** that are widely used in industry but rarely taught in courses. Each model includes real-world datasets, hyperparameter tuning, and deployment-ready pipelines.

## Models Implemented

| Model | Notebook | Use Case |
|---|---|---|
| Local Outlier Factor (LOF) | `01_anomaly_detection.ipynb` | Fraud detection, sensor anomalies |
| Isolation Forest | `01_anomaly_detection.ipynb` | Large-scale anomaly detection |
| Robust Random Cut Forest (RRCF) | `01_anomaly_detection.ipynb` | Real-time streaming anomaly detection |
| Gradient Boosting (XGBoost/LightGBM/CatBoost) | `02_gradient_boosting.ipynb` | Credit scoring, forecasting, marketing |
| Gaussian Processes | `03_gaussian_processes.ipynb` | Uncertainty estimation, Bayesian optimization |

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook
```

## Datasets

- Credit Card Fraud Detection (auto-downloaded)
- Synthetic streaming sensor data (generated in-notebook)
- UCI ML Repository datasets (auto-downloaded)

## Project Structure

```
03_Production_ML_Models/
├── 01_anomaly_detection.ipynb      # LOF, Isolation Forest, RRCF
├── 02_gradient_boosting.ipynb      # XGBoost, LightGBM, CatBoost
├── 03_gaussian_processes.ipynb     # GPyTorch, scikit-learn GP
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/                           # Auto-populated datasets
├── outputs/                        # Model artifacts & plots
└── docs/                           # Architecture docs
```

## License

MIT
