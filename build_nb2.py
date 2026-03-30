"""Build Script: Gradient Boosting notebook (XGBoost, LightGBM, CatBoost)"""
import json, os

def make_cell(ct, src):
    c = {"cell_type": ct, "metadata": {}, "source": src.split("\n") if isinstance(src, str) else src}
    if ct == "code": c["execution_count"] = None; c["outputs"] = []
    lines = c["source"]
    c["source"] = [l + "\n" if i < len(lines)-1 else l for i, l in enumerate(lines)]
    return c

def make_nb(cells):
    return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.0"}},"cells":cells}

BASE = r"D:\Completed Projects\03_Production_ML_Models"

cells = [
    make_cell("markdown", """# Gradient Boosting Machines for Production
## XGBoost | LightGBM | CatBoost

**Industry Use Cases:** Credit scoring, churn prediction, marketing attribution, demand forecasting

These routinely outperform deep learning on tabular/structured business data.

---"""),

    make_cell("code", """# CELL 1: Install & Import
import subprocess, sys
for pkg in ["numpy","pandas","scikit-learn","matplotlib","seaborn","xgboost","lightgbm","catboost","optuna","shap","tqdm"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import xgboost as xgb, lightgbm as lgb, catboost as cb
import optuna, time, joblib, warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
    average_precision_score, roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.datasets import fetch_openml

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
SEED = 42; np.random.seed(SEED)
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
print("All imports ready!")"""),

    make_cell("markdown", """## Section 1: Dataset — Credit Default Prediction
Using the UCI Default of Credit Card Clients dataset (30,000 clients, 23 features)."""),

    make_cell("code", """# CELL 2: Load Dataset
try:
    # Try OpenML first
    data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
    df = data.frame
    target_col = df.columns[-1]
except:
    # Fallback: generate synthetic credit data
    print("Generating synthetic credit scoring dataset...")
    n = 10000
    np.random.seed(SEED)
    df = pd.DataFrame({
        'credit_limit': np.random.lognormal(10, 1, n),
        'age': np.random.randint(21, 65, n),
        'education': np.random.choice([1,2,3,4], n),
        'marriage': np.random.choice([1,2,3], n),
        'pay_status_1': np.random.choice([-1,0,1,2,3], n, p=[0.3,0.4,0.15,0.1,0.05]),
        'pay_status_2': np.random.choice([-1,0,1,2,3], n, p=[0.3,0.4,0.15,0.1,0.05]),
        'bill_amt_1': np.random.lognormal(8, 2, n),
        'bill_amt_2': np.random.lognormal(8, 2, n),
        'pay_amt_1': np.random.lognormal(7, 2, n),
        'pay_amt_2': np.random.lognormal(7, 2, n),
    })
    # Create target with realistic default rate (~22%)
    proba = 1 / (1 + np.exp(-(df['pay_status_1']*0.5 + df['pay_status_2']*0.3 - 1)))
    df['default'] = (np.random.random(n) < proba).astype(int)
    target_col = 'default'

print(f"Dataset: {df.shape}")
print(f"Default rate: {df[target_col].mean()*100:.1f}%")
df.head()"""),

    make_cell("code", """# CELL 3: Preprocessing
# Encode categoricals if needed
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(target_col, axis=1)
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Default rate - Train: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")"""),

    make_cell("markdown", """## Section 2: XGBoost with Optuna Tuning"""),

    make_cell("code", """# CELL 4: XGBoost
def xgb_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': (y_train==0).sum() / (y_train==1).sum(),
        'random_state': SEED, 'eval_metric': 'auc', 'use_label_encoder': False
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

print("Tuning XGBoost (15 trials)...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(xgb_objective, n_trials=15, show_progress_bar=True)

xgb_model = xgb.XGBClassifier(**study_xgb.best_params, random_state=SEED, eval_metric='auc', use_label_encoder=False)
xgb_model.fit(X_train, y_train)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict(X_test)
print(f"\\nXGBoost AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
print(f"XGBoost F1: {f1_score(y_test, xgb_pred):.4f}")"""),

    make_cell("markdown", """## Section 3: LightGBM"""),

    make_cell("code", """# CELL 5: LightGBM
def lgb_objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'is_unbalance': True, 'random_state': SEED, 'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

print("Tuning LightGBM (15 trials)...")
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(lgb_objective, n_trials=15, show_progress_bar=True)

lgb_model = lgb.LGBMClassifier(**study_lgb.best_params, random_state=SEED, verbose=-1)
lgb_model.fit(X_train, y_train)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred = lgb_model.predict(X_test)
print(f"\\nLightGBM AUC: {roc_auc_score(y_test, lgb_proba):.4f}")
print(f"LightGBM F1: {f1_score(y_test, lgb_pred):.4f}")"""),

    make_cell("markdown", """## Section 4: CatBoost"""),

    make_cell("code", """# CELL 6: CatBoost
def cb_objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'auto_class_weights': 'Balanced',
        'random_seed': SEED, 'verbose': 0
    }
    model = cb.CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

print("Tuning CatBoost (10 trials)...")
study_cb = optuna.create_study(direction='maximize')
study_cb.optimize(cb_objective, n_trials=10, show_progress_bar=True)

cb_model = cb.CatBoostClassifier(**study_cb.best_params, random_seed=SEED, verbose=0)
cb_model.fit(X_train, y_train)
cb_proba = cb_model.predict_proba(X_test)[:, 1]
cb_pred = cb_model.predict(X_test)
print(f"\\nCatBoost AUC: {roc_auc_score(y_test, cb_proba):.4f}")
print(f"CatBoost F1: {f1_score(y_test, cb_pred):.4f}")"""),

    make_cell("code", """# CELL 7: Comparison Dashboard
models_data = {
    'XGBoost': {'proba': xgb_proba, 'pred': xgb_pred, 'model': xgb_model},
    'LightGBM': {'proba': lgb_proba, 'pred': lgb_pred, 'model': lgb_model},
    'CatBoost': {'proba': cb_proba, 'pred': cb_pred, 'model': cb_model}
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curves
for name, d in models_data.items():
    fpr, tpr, _ = roc_curve(y_test, d['proba'])
    auc = roc_auc_score(y_test, d['proba'])
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2)
axes[0].plot([0,1],[0,1],'k--',alpha=0.3); axes[0].set_title('ROC Curves'); axes[0].legend()

# Feature Importance (XGBoost)
imp = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
imp.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Top 10 Features (XGBoost)')

# Metrics bar chart
metrics = []
for name, d in models_data.items():
    metrics.append({'Model': name, 'AUC': roc_auc_score(y_test, d['proba']),
                    'F1': f1_score(y_test, d['pred']), 'AP': average_precision_score(y_test, d['proba'])})
mdf = pd.DataFrame(metrics)
mdf.set_index('Model')[['AUC','F1','AP']].plot(kind='bar', ax=axes[2], rot=0)
axes[2].set_title('Model Metrics Comparison'); axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gbm_comparison.png", dpi=150, bbox_inches='tight')
plt.show()"""),

    make_cell("code", """# CELL 8: Save Models
for name, d in models_data.items():
    joblib.dump(d['model'], OUTPUT_DIR / f"{name.lower()}_model.pkl")
    print(f"Saved: {name.lower()}_model.pkl")

# Save comparison
mdf.to_csv(OUTPUT_DIR / "gbm_comparison.csv", index=False)

print("\\n" + "="*50)
print("  GRADIENT BOOSTING - COMPLETE")
print("="*50)
print(mdf.to_string(index=False))
print("\\nKey Takeaways:")
print("  - LightGBM: Fastest training, great for large datasets")
print("  - XGBoost: Most mature, best regularization options")
print("  - CatBoost: Best default performance, handles categoricals natively")"""),
]

nb = make_nb(cells)
with open(os.path.join(BASE, "02_gradient_boosting.ipynb"), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Created: 02_gradient_boosting.ipynb")
