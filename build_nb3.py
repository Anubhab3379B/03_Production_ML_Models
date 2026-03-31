"""Build Script: Gaussian Processes notebook"""
import json, os

def mc(ct, src):
    c = {"cell_type": ct, "metadata": {}, "source": src.split("\n")}
    if ct == "code": c["execution_count"] = None; c["outputs"] = []
    c["source"] = [l + "\n" if i < len(c["source"])-1 else l for i, l in enumerate(c["source"])]
    return c

BASE = r"D:\Completed Projects\03_Production_ML_Models"
cells = [
    mc("markdown", "# Gaussian Processes for Uncertainty-Aware Predictions\n## scikit-learn GP | GPyTorch Deep GP | Bayesian Optimization\n\n**Use Cases:** Scientific modeling, forecasting with confidence intervals, Bayesian hyperparameter optimization\n\n---"),

    mc("code", "# CELL 1: Setup\nimport subprocess, sys\nfor p in ['numpy','pandas','scikit-learn','matplotlib','gpytorch','torch','plotly','optuna']:\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])\n\nimport numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, gpytorch, warnings, joblib\nfrom pathlib import Path\nfrom sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier\nfrom sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, RationalQuadratic\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\nfrom sklearn.preprocessing import StandardScaler\nwarnings.filterwarnings('ignore')\nSEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)\nOUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)\nprint('Ready!')"),

    mc("markdown", "## Section 1: scikit-learn Gaussian Process Regression\nGP provides a **mean prediction + uncertainty band** — crucial for risk-sensitive applications."),

    mc("code", """# CELL 2: Generate regression data with known uncertainty
# Simulate sensor calibration data (nonlinear + heteroscedastic noise)
X_full = np.sort(np.random.uniform(-5, 5, 200)).reshape(-1, 1)
y_full = np.sin(X_full).ravel() + 0.5 * np.cos(2*X_full).ravel()
y_full += np.random.normal(0, 0.1 + 0.1*np.abs(X_full.ravel()), len(X_full))

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=SEED)

# Composite kernel: captures signal + noise
kernel = (ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) 
          + WhiteKernel(noise_level=0.1))

# Fit GP
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=SEED, normalize_y=True)
gp.fit(X_train, y_train)

# Predict with uncertainty
X_plot = np.linspace(-6, 6, 300).reshape(-1, 1)
y_mean, y_std = gp.predict(X_plot, return_std=True)
y_pred = gp.predict(X_test)

# Metrics
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Learned kernel: {gp.kernel_}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X_train, y_train, c='steelblue', s=20, alpha=0.5, label='Train')
ax.scatter(X_test, y_test, c='crimson', s=20, alpha=0.5, label='Test')
ax.plot(X_plot, y_mean, 'k-', linewidth=2, label='GP Mean')
ax.fill_between(X_plot.ravel(), y_mean - 2*y_std, y_mean + 2*y_std, alpha=0.2, color='steelblue', label='95% CI')
ax.set_title('Gaussian Process Regression with Uncertainty')
ax.legend(); ax.set_xlabel('X'); ax.set_ylabel('y')
plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'gp_regression.png', dpi=150); plt.show()
print('Saved: gp_regression.png')"""),

    mc("markdown", "## Section 2: GPyTorch — Scalable Exact GP\nGPyTorch leverages GPU acceleration and can handle 10,000+ data points efficiently."),

    mc("code", """# CELL 3: GPyTorch Exact GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    \"\"\"Exact GP with Matern kernel + automatic relevance determination.\"\"\"
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Prepare data
train_x = torch.FloatTensor(X_train)
train_y = torch.FloatTensor(y_train)
test_x = torch.FloatTensor(X_test)

# Initialize
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Training
model.train(); likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    if (i+1) % 25 == 0:
        print(f'Iter {i+1}/100 - Loss: {loss.item():.3f}')

# Prediction
model.eval(); likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(test_x))
    gpy_mean = pred.mean.numpy()
    gpy_std = pred.stddev.numpy()

print(f"\\nGPyTorch R2: {r2_score(y_test, gpy_mean):.4f}")
print(f"GPyTorch RMSE: {np.sqrt(mean_squared_error(y_test, gpy_mean)):.4f}")"""),

    mc("markdown", "## Section 3: Bayesian Optimization with GP\nUsing GP as a surrogate model to optimize expensive black-box functions."),

    mc("code", """# CELL 4: Bayesian Optimization from scratch
def objective_function(x):
    \"\"\"Expensive black-box function to optimize (Hartmann-like).\"\"\"
    return -(np.sin(3*x) * np.cos(x) + np.sin(x**2) * 0.5)

def bayesian_optimization(func, bounds, n_init=5, n_iter=20, seed=42):
    \"\"\"GP-based Bayesian optimization with Expected Improvement.\"\"\"
    np.random.seed(seed)
    # Initial random samples
    X_samples = np.random.uniform(bounds[0], bounds[1], n_init).reshape(-1, 1)
    y_samples = np.array([func(x) for x in X_samples.ravel()])
    
    history = {'X': list(X_samples.ravel()), 'y': list(y_samples), 'best_y': [y_samples.min()]}
    
    for i in range(n_iter):
        # Fit GP surrogate
        kernel = ConstantKernel() * Matern(nu=2.5) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=seed)
        gp.fit(X_samples, y_samples)
        
        # Expected Improvement acquisition
        X_cand = np.linspace(bounds[0], bounds[1], 1000).reshape(-1, 1)
        mu, sigma = gp.predict(X_cand, return_std=True)
        best_y = y_samples.min()
        
        from scipy.stats import norm
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (best_y - mu) / sigma
            ei = (best_y - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0] = 0
        
        # Select next point
        next_x = X_cand[np.argmax(ei)]
        next_y = func(next_x[0])
        
        X_samples = np.vstack([X_samples, next_x.reshape(1, -1)])
        y_samples = np.append(y_samples, next_y)
        history['X'].append(next_x[0])
        history['y'].append(next_y)
        history['best_y'].append(min(history['best_y'][-1], next_y))
    
    return history, gp

# Run optimization
history, final_gp = bayesian_optimization(objective_function, bounds=(-3, 3), n_init=5, n_iter=25)

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['best_y'], 'bo-', linewidth=2)
axes[0].set_title('Bayesian Optimization Convergence')
axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Best Objective')

X_plot = np.linspace(-3, 3, 500).reshape(-1, 1)
axes[1].plot(X_plot, [objective_function(x) for x in X_plot.ravel()], 'k-', label='True Function')
axes[1].scatter(history['X'][:5], history['y'][:5], c='blue', s=50, zorder=5, label='Initial')
axes[1].scatter(history['X'][5:], history['y'][5:], c='red', s=50, zorder=5, label='BO Points')
axes[1].set_title('Sampling Pattern'); axes[1].legend()

plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'bayesian_optimization.png', dpi=150); plt.show()

print(f"\\nBest value found: {min(history['y']):.6f}")
print(f"At x = {history['X'][np.argmin(history['y'])]:.4f}")"""),

    mc("code", """# CELL 5: Save all models
joblib.dump(gp, OUTPUT_DIR / 'sklearn_gp_model.pkl')
torch.save(model.state_dict(), OUTPUT_DIR / 'gpytorch_model.pt')

print("="*50)
print("  GAUSSIAN PROCESSES - COMPLETE")
print("="*50)
print("Outputs:")
print("  - sklearn_gp_model.pkl")
print("  - gpytorch_model.pt")
print("  - gp_regression.png")
print("  - bayesian_optimization.png")
print("\\nProject P3 fully complete! All 3 notebooks built.")"""),
]

nb = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.0"}},"cells":cells}
with open(os.path.join(BASE, "03_gaussian_processes.ipynb"), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Created: 03_gaussian_processes.ipynb")
