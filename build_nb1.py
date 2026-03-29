"""
Build Script: Generates all Jupyter notebooks for Project 03 - Production ML Models
Run this script to create the notebooks programmatically.
"""
import json, os

def make_cell(cell_type, source):
    """Create a notebook cell."""
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.split("\n") if isinstance(source, str) else source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    # Fix: each line needs \n except last
    lines = cell["source"]
    cell["source"] = [l + "\n" if i < len(lines)-1 else l for i, l in enumerate(lines)]
    return cell

def make_notebook(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    }, "cells": cells}

def save_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created: {path}")

BASE = r"D:\Completed Projects\03_Production_ML_Models"

# ============================================================
# NOTEBOOK 1: ANOMALY DETECTION
# ============================================================
print("Building 01_anomaly_detection.ipynb...")
cells = [
    make_cell("markdown", """# 🔍 Production Anomaly Detection Models
## Local Outlier Factor | Isolation Forest | Robust Random Cut Forest

**Industry Use Cases:** Fraud detection, sensor monitoring, network intrusion, IoT anomalies

This notebook implements three production-grade anomaly detection models with:
- Real credit card fraud dataset (auto-downloaded)
- Streaming anomaly detection simulation
- Model comparison & hyperparameter tuning with Optuna
- Deployment-ready pipelines

---"""),

    make_cell("code", """# ============================================================
# CELL 1: Environment Setup & Dependencies
# ============================================================
# Install all required packages (run once)
import subprocess, sys

packages = [
    "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn",
    "plotly", "rrcf", "optuna", "tqdm", "requests", "joblib", "ipywidgets"
]
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ All dependencies installed successfully!")"""),

    make_cell("code", """# ============================================================
# CELL 2: Imports & Configuration
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, os, time, json, hashlib
from pathlib import Path

# Scikit-learn models
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, roc_curve)
from sklearn.pipeline import Pipeline

# RRCF for streaming
import rrcf

# Hyperparameter tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Utilities
from tqdm.auto import tqdm
import joblib, requests

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Project paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)

print("✅ All imports loaded | Paths configured | Seed set to", SEED)"""),

    make_cell("markdown", """## 📊 Section 1: Dataset Loading & Exploration
We use the **Credit Card Fraud Detection** dataset — 284,807 transactions with 492 frauds (0.17%).
This extreme class imbalance is exactly what production anomaly detectors face."""),

    make_cell("code", """# ============================================================
# CELL 3: Auto-Download Credit Card Fraud Dataset
# ============================================================
DATASET_PATH = DATA_DIR / "creditcard.csv"

def download_dataset():
    \"\"\"Download credit card fraud dataset from multiple mirrors.\"\"\"
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
    ]
    for url in urls:
        try:
            print(f"Downloading from {url[:60]}...")
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(DATASET_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {DATASET_PATH} ({DATASET_PATH.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"  ⚠️ Failed: {e}")
    return False

# Download if not cached
if not DATASET_PATH.exists():
    if not download_dataset():
        print("❌ Auto-download failed. Generating synthetic fraud dataset instead...")
        # Generate synthetic dataset as fallback
        n_normal, n_fraud = 10000, 50
        normal = np.random.randn(n_normal, 30) * 0.5
        fraud = np.random.randn(n_fraud, 30) * 2 + np.random.choice([-3, 3], size=(n_fraud, 30))
        X = np.vstack([normal, fraud])
        y = np.array([0]*n_normal + [1]*n_fraud)
        cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        df = pd.DataFrame(X, columns=cols)
        df['Class'] = y
        df.to_csv(DATASET_PATH, index=False)
        print(f"✅ Generated synthetic dataset: {len(df)} rows")
else:
    print(f"✅ Dataset already cached: {DATASET_PATH}")

# Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"\\nDataset shape: {df.shape}")
print(f"Fraud ratio: {df['Class'].mean()*100:.3f}%")
print(f"Normal: {(df['Class']==0).sum():,} | Fraud: {(df['Class']==1).sum():,}")
df.head()"""),

    make_cell("code", """# ============================================================
# CELL 4: Exploratory Data Analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Class distribution
df['Class'].value_counts().plot(kind='bar', ax=axes[0,0], color=['steelblue', 'crimson'])
axes[0,0].set_title('Class Distribution (0=Normal, 1=Fraud)')
axes[0,0].set_ylabel('Count')

# 2. Transaction amount distribution
axes[0,1].hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, label='Normal', color='steelblue')
axes[0,1].hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, label='Fraud', color='crimson')
axes[0,1].set_title('Transaction Amount Distribution')
axes[0,1].legend()
axes[0,1].set_xlim(0, 500)

# 3. Correlation of top features with fraud
feature_cols = [c for c in df.columns if c.startswith('V')]
corrs = df[feature_cols + ['Class']].corr()['Class'].drop('Class').abs().sort_values(ascending=False)
corrs.head(10).plot(kind='barh', ax=axes[1,0], color='darkorange')
axes[1,0].set_title('Top 10 Features Correlated with Fraud')

# 4. Time distribution
axes[1,1].hist(df[df['Class']==0]['Time'], bins=50, alpha=0.7, label='Normal', color='steelblue')
axes[1,1].hist(df[df['Class']==1]['Time'], bins=50, alpha=0.7, label='Fraud', color='crimson')
axes[1,1].set_title('Transaction Time Distribution')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_overview.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plots saved to outputs/eda_overview.png")"""),

    make_cell("code", """# ============================================================
# CELL 5: Data Preprocessing Pipeline
# ============================================================
# Separate features and target
X = df.drop('Class', axis=1).values
y = df['Class'].values

# Robust scaling (handles outliers better than StandardScaler)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratification (preserves fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=SEED, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples ({y_train.mean()*100:.3f}% fraud)")
print(f"Test set:     {X_test.shape[0]:,} samples ({y_test.mean()*100:.3f}% fraud)")
print("✅ Data preprocessed with RobustScaler + stratified split")"""),

    make_cell("markdown", """## 🔬 Section 2: Local Outlier Factor (LOF)
**How it works:** Compares each point's local density to its neighbors. Points in sparser regions get higher anomaly scores.

**Best for:** Detecting subtle anomalies that differ from their local neighborhood — fraud patterns that look normal globally but unusual locally."""),

    make_cell("code", """# ============================================================
# CELL 6: Local Outlier Factor — Training & Evaluation
# ============================================================
def evaluate_model(y_true, y_pred, y_scores, model_name):
    \"\"\"Comprehensive evaluation of anomaly detection model.\"\"\"
    # Convert LOF convention: -1 = anomaly → 1, 1 = normal → 0
    if set(np.unique(y_pred)) == {-1, 1}:
        y_pred = (y_pred == -1).astype(int)

    print(f"\\n{'='*60}")
    print(f"  {model_name} — Results")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred)
    print(f"ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f} | F1: {f1:.4f}")
    return {'model': model_name, 'auc': auc, 'ap': ap, 'f1': f1, 'y_pred': y_pred, 'y_scores': y_scores}

# LOF — novelty detection mode
# contamination = expected fraud ratio
contamination = y_train.mean()

lof = LocalOutlierFactor(
    n_neighbors=20,        # neighborhood size
    contamination=contamination,
    novelty=True,          # enables predict on new data
    metric='minkowski',
    n_jobs=-1              # parallelize
)

print("Training LOF...")
t0 = time.time()
lof.fit(X_train)
lof_preds = lof.predict(X_test)
lof_scores = -lof.decision_function(X_test)  # negate: higher = more anomalous
print(f"LOF training + inference: {time.time()-t0:.2f}s")

lof_results = evaluate_model(y_test, lof_preds, lof_scores, "Local Outlier Factor")"""),

    make_cell("markdown", """## 🌲 Section 3: Isolation Forest
**How it works:** Randomly partitions data with trees. Anomalies need fewer splits to isolate → shorter path length = higher anomaly score.

**Best for:** High-dimensional datasets, large-scale anomaly detection. Industry workhorse."""),

    make_cell("code", """# ============================================================
# CELL 7: Isolation Forest — Training & Optuna Tuning
# ============================================================
def objective_iforest(trial):
    \"\"\"Optuna objective for Isolation Forest hyperparameter tuning.\"\"\"
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'contamination': trial.suggest_float('contamination', 0.001, 0.01),
        'random_state': SEED, 'n_jobs': -1
    }
    model = IsolationForest(**params)
    model.fit(X_train)
    scores = -model.decision_function(X_test)
    return average_precision_score(y_test, scores)

# Run Optuna tuning (20 trials for speed)
print("🔧 Tuning Isolation Forest with Optuna (20 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective_iforest, n_trials=20, show_progress_bar=True)

print(f"\\nBest Average Precision: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train with best params
best_if = IsolationForest(**study.best_params, random_state=SEED, n_jobs=-1)
t0 = time.time()
best_if.fit(X_train)
if_preds = best_if.predict(X_test)
if_scores = -best_if.decision_function(X_test)
print(f"\\nIsolation Forest training + inference: {time.time()-t0:.2f}s")

if_results = evaluate_model(y_test, if_preds, if_scores, "Isolation Forest (Tuned)")"""),

    make_cell("markdown", """## 🌊 Section 4: Robust Random Cut Forest (RRCF)
**How it works:** Online/streaming algorithm. Builds random cut trees and measures how much removing a point changes the tree structure (CoDisplacement score).

**Best for:** Real-time streaming data, IoT monitoring, live fraud detection. Adapts to concept drift."""),

    make_cell("code", """# ============================================================
# CELL 8: RRCF — Streaming Anomaly Detection
# ============================================================
# RRCF works on streaming data — we simulate a real-time stream
# Using a subset for speed (RRCF is O(n) per point per tree)

STREAM_SIZE = min(5000, len(X_test))
NUM_TREES = 100
TREE_SIZE = 256

print(f"🌊 Running RRCF Streaming Detection on {STREAM_SIZE} points...")
print(f"   Trees: {NUM_TREES} | Tree size: {TREE_SIZE}")

# Build forest
forest = []
for _ in range(NUM_TREES):
    tree = rrcf.RCTree()
    forest.append(tree)

# Stream points and compute anomaly scores
avg_codisp = {}
stream_data = X_test[:STREAM_SIZE]

for idx in tqdm(range(STREAM_SIZE), desc="Streaming"):
    point = stream_data[idx]
    for tree in forest:
        # If tree is full, drop oldest point
        if len(tree.leaves) > TREE_SIZE:
            oldest = min(tree.leaves.keys())
            tree.forget_point(oldest)
        # Insert new point
        tree.insert_point(point, index=idx)
        # Compute CoDisplacement
        if idx not in avg_codisp:
            avg_codisp[idx] = 0
        avg_codisp[idx] += tree.codisp(idx) / NUM_TREES

# Convert to arrays
rrcf_scores = np.array([avg_codisp[i] for i in range(STREAM_SIZE)])
rrcf_y_test = y_test[:STREAM_SIZE]

# Threshold: top percentile based on contamination
threshold = np.percentile(rrcf_scores, 100 * (1 - contamination))
rrcf_preds = (rrcf_scores >= threshold).astype(int)

rrcf_results = evaluate_model(rrcf_y_test, rrcf_preds, rrcf_scores, "RRCF (Streaming)")"""),

    make_cell("code", """# ============================================================
# CELL 9: Model Comparison Dashboard
# ============================================================
results = [lof_results, if_results, rrcf_results]
comparison_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_pred' and k != 'y_scores'} for r in results])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Metric comparison
metrics = ['auc', 'ap', 'f1']
x = np.arange(len(results))
width = 0.25
for i, metric in enumerate(metrics):
    axes[0].bar(x + i*width, comparison_df[metric], width, label=metric.upper())
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(comparison_df['model'], rotation=15, ha='right')
axes[0].legend()
axes[0].set_title('Model Comparison')
axes[0].set_ylim(0, 1)

# 2. ROC Curves
for r in results:
    if len(r['y_scores']) == len(y_test):
        fpr, tpr, _ = roc_curve(y_test, r['y_scores'])
    else:
        fpr, tpr, _ = roc_curve(rrcf_y_test, r['y_scores'])
    axes[1].plot(fpr, tpr, label=f"{r['model']} (AUC={r['auc']:.3f})")
axes[1].plot([0,1], [0,1], 'k--', alpha=0.5)
axes[1].set_title('ROC Curves')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend(fontsize=8)

# 3. Precision-Recall Curves
for r in results:
    if len(r['y_scores']) == len(y_test):
        prec, rec, _ = precision_recall_curve(y_test, r['y_scores'])
    else:
        prec, rec, _ = precision_recall_curve(rrcf_y_test, r['y_scores'])
    axes[2].plot(rec, prec, label=f"{r['model']} (AP={r['ap']:.3f})")
axes[2].set_title('Precision-Recall Curves')
axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Comparison saved to outputs/model_comparison.png")"""),

    make_cell("code", """# ============================================================
# CELL 10: Save Models & Final Summary
# ============================================================
# Save trained models for deployment
joblib.dump(lof, OUTPUT_DIR / "lof_model.pkl")
joblib.dump(best_if, OUTPUT_DIR / "isolation_forest_model.pkl")
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

# Save comparison results
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\\n" + "="*60)
print("  📊 FINAL SUMMARY — Anomaly Detection Models")
print("="*60)
print(comparison_df.to_string(index=False))
print(f"\\n✅ Models saved to {OUTPUT_DIR}/")
print("✅ Ready for production deployment!")
print("\\n🔑 Key Takeaways:")
print("  • Isolation Forest: Best overall for batch processing")
print("  • LOF: Best for local/neighborhood-based anomalies")
print("  • RRCF: Best for real-time streaming applications")"""),

    make_cell("markdown", """---
## ✅ Notebook Complete

**What was built:**
1. **Local Outlier Factor** — neighborhood-based anomaly detection
2. **Isolation Forest** — with Optuna hyperparameter tuning
3. **RRCF** — real-time streaming anomaly detection

**Outputs saved:**
- `outputs/lof_model.pkl` — trained LOF model
- `outputs/isolation_forest_model.pkl` — tuned Isolation Forest
- `outputs/scaler.pkl` — fitted RobustScaler
- `outputs/model_comparison.csv` — metrics comparison
- `outputs/model_comparison.png` — visual comparison

**Next:** See `02_gradient_boosting.ipynb` for XGBoost, LightGBM, CatBoost""")
]

nb1 = make_notebook(cells)
save_nb(nb1, os.path.join(BASE, "01_anomaly_detection.ipynb"))

print("\n✅ Notebook 1 complete!")
print("Run: jupyter notebook '01_anomaly_detection.ipynb'")
