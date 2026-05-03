"""
# Statistics Toolkit — Plots

Diagram helpers for the statistics half of the course.

| Function | What it draws |
|---|---|
| `plot_histogram(data)` | Absolute, relative, and normalized histograms |
| `plot_running_stats(data)` | Running sample mean x̄ᵢ and x̄ᵢ ± Sᵢ as i grows |
| `plot_loglikelihood(data, n, sigma2)` | Log-likelihood function for μ given the first n observations |
| `plot_regression(x, y, fit)` | Scatter plot with optional fitted regression line (default: on) |
| `plot_residuals(x, y)` | Standardized residuals against x with ±1 and ±2 reference bands |
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CSV LOADING
# =============================================================================

def load_csv(filepath, col, header=True):
    """
    Loads a single column from a CSV file as a numpy array.

    **Parameters**

    - `filepath` *(str)* — path to the CSV file, relative to your notebook location.
    - `col` *(str or int)* — column name if the CSV has a header row, or 0-indexed
      integer if it does not.
    - `header` *(bool, optional)* — whether the CSV has a header row; default `True`.

    **Examples**

    ```python
    data = load_csv("temperature.csv", col="temperature")
    data = load_csv("temperature.csv", col=1, header=False)  # second column, no header
    ```
    """
    df = pd.read_csv("input_data/" + filepath, header=0 if header else None)
    if header:
        df.columns = df.columns.str.strip()
    return df[col].values

# =============================================================================
# 1. HISTOGRAM (absolute, relative, normalized)
# =============================================================================

def plot_histogram(data, bins=30, title="Histogram"):
    """
    Plots absolute, relative, and normalized histograms side by side.

    - **Absolute**: raw counts per bin.
    - **Relative**: counts divided by total n; bars sum to 1.
    - **Normalized (density)**: scaled so the total *area* equals 1,
      making it comparable to a PDF.

    **Parameters**

    - `data` *(list or numpy array)* — the raw observations.
    - `bins` *(int, optional)* — number of histogram bins; default 30.
    - `title` *(str, optional)* — base title; variant name is appended automatically.

    **Examples**

    ```python
    data = load_csv("latency.csv", col="latency")
    plot_histogram(data)

    data = load_csv("temperature.csv", col=1, header=False)
    plot_histogram(data, bins=20, title="Room Temperature")
    ```
    """
    data = np.asarray(data)
    n = len(data)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Absolute
    axes[0].hist(data, bins=bins, color='C0', edgecolor='white')
    axes[0].set_title(f"{title} — Absolute")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Count")

    # Relative
    axes[1].hist(data, bins=bins, weights=np.ones(n) / n,
                 color='C0', edgecolor='white')
    axes[1].set_title(f"{title} — Relative")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Relative frequency")

    # Normalized (density)
    axes[2].hist(data, bins=bins, density=True, color='C0', edgecolor='white')
    axes[2].set_title(f"{title} — Normalized (density)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. RUNNING SAMPLE MEAN AND VARIANCE
# =============================================================================

def plot_running_stats(data, title="Running Sample Mean"):
    """
    Plots the running sample mean x̄ᵢ and the band x̄ᵢ ± Sᵢ as i grows
    from 2 to n (variance requires at least 2 observations).

    Useful for visualising convergence of the sample mean and the
    law of large numbers in action.

    **Parameters**

    - `data` *(list or numpy array)* — the raw observations, in the order they were collected.
    - `title` *(str, optional)* — plot title; default `"Running Sample Mean"`.

    **Examples**

    ```python
    plot_running_stats(dice_rolls)

    data = load_csv("temperature.csv", col=1, header=False)
    plot_running_stats(data, title="Temperature Convergence")
    ```
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    indices = np.arange(2, n + 1)

    means = np.array([np.mean(data[:i]) for i in indices])
    stds  = np.array([np.std(data[:i], ddof=1) for i in indices])

    plt.figure()
    plt.plot(indices, means, color='C0', linewidth=2, label=r"$\bar{X}_i$")
    plt.fill_between(indices,
                     means - stds,
                     means + stds,
                     alpha=0.25, color='C0',
                     label=r"$\bar{X}_i \pm S_i$")
    plt.xlabel("i (number of observations)")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. LOG-LIKELIHOOD FOR μ (Gaussian model, known σ²)
# =============================================================================

def plot_loglikelihood(data, n, sigma2, title="Log-Likelihood for μ"):
    """
    Plots the log-likelihood function ℓ(μ) for the mean of a Gaussian model,
    using the first `n` observations and treating σ² as known.

    The log-likelihood for μ given X₁, …, Xₙ ~ N(μ, σ²) is:

        ℓ(μ) = −n/2 · log(2πσ²) − 1/(2σ²) · Σ(Xᵢ − μ)²

    A vertical dashed line marks the MLE, which equals the sample mean x̄.

    > **Note:** σ² should be fixed at your assumed or estimated value
    > (e.g. the sample variance from the full dataset), not varied.

    **Parameters**

    - `data` *(list or numpy array)* — the full dataset; only the first `n` values are used.
    - `n` *(int)* — number of observations to use.
    - `sigma2` *(float)* — assumed known variance σ² — **NOT standard deviation**.
    - `title` *(str, optional)* — plot title; default `"Log-Likelihood for μ"`.

    **Examples**

    ```python
    from stats_compute import load_csv, sample_stats

    data = load_csv("temperature.csv", col=1, header=False)
    _, _, sigma2, _ = sample_stats(data)    # estimate sigma2 from full dataset

    plot_loglikelihood(data, n=10,  sigma2=sigma2)
    plot_loglikelihood(data, n=100, sigma2=sigma2)
    ```
    """
    data = np.asarray(data, dtype=float)
    subset = data[:n]
    x_bar = np.mean(subset)

    # Range of μ values to evaluate: centre on x̄, width scales with σ/√n
    sigma = np.sqrt(sigma2)
    mu_range = np.linspace(x_bar - 4 * sigma / np.sqrt(n),
                           x_bar + 4 * sigma / np.sqrt(n),
                           500)

    def log_likelihood(mu):
        return (-n / 2 * np.log(2 * np.pi * sigma2)
                - 1 / (2 * sigma2) * np.sum((subset - mu) ** 2))

    ll_values = np.array([log_likelihood(mu) for mu in mu_range])

    plt.figure()
    plt.plot(mu_range, ll_values, color='C0', linewidth=2)
    plt.axvline(x_bar, color='C1', linestyle='--',
                label=f"MLE = x̄ = {x_bar:.4f}")
    plt.xlabel("μ")
    plt.ylabel("ℓ(μ)")
    plt.title(f"{title} (n = {n})")
    plt.legend()
    plt.tight_layout()
    plt.show()



# =============================================================================
# 4. REGRESSION SCATTER PLOT WITH OPTIONAL FITTED LINE
# =============================================================================

def plot_regression(x, y, fit=True, xlabel="x", ylabel="y", title="Scatter Plot"):
    """
    Scatter plot of (x, y) data with an optional fitted regression line.

    The line is always labelled with the fitted equation ŷ = β̂₀ + β̂₁x,
    so it also serves as a quick visual check of your hand-computed coefficients.
    Set `fit=False` for a plain scatter when you just want to assess whether
    a linear relationship looks reasonable before committing to a model.

    **Parameters**

    - `x` *(list or numpy array)* — predictor values.
    - `y` *(list or numpy array)* — response values.
    - `fit` *(bool, optional)* — whether to overlay the fitted line; default `True`.
    - `xlabel` *(str, optional)* — x-axis label; default `"x"`.
    - `ylabel` *(str, optional)* — y-axis label; default `"y"`.
    - `title` *(str, optional)* — plot title; default `"Scatter Plot"`.

    **Examples**

    ```python
    x = [0.41, 0.46, 0.44, 0.47, 0.42]
    y = [1850, 2620, 2340, 2690, 2160]

    # Scatter + fitted line (default)
    plot_regression(x, y, xlabel="Specific gravity", ylabel="Crushing strength (psi)",
                    title="Wood Strength vs. Specific Gravity")

    # Plain scatter only — assess linearity first
    plot_regression(x, y, fit=False, title="Wood Strength — raw scatter")
    ```
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    plt.figure()
    plt.scatter(x, y, color='C0', zorder=3, label="Data")

    if fit:
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        s_xx  = np.sum((x - x_bar) ** 2)
        s_xy  = np.sum((x - x_bar) * (y - y_bar))
        beta1 = s_xy / s_xx
        beta0 = y_bar - beta1 * x_bar

        x_line = np.linspace(x.min(), x.max(), 300)
        y_line = beta0 + beta1 * x_line
        sign   = "+" if beta1 >= 0 else "-"
        plt.plot(x_line, y_line, color='C1', linewidth=2,
                 label=f"ŷ = {beta0:.4f} {sign} {abs(beta1):.4f}x")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. STANDARDIZED RESIDUAL PLOT
# =============================================================================

def plot_residuals(x, y, xlabel="x", title="Standardized Residuals"):
    """
    Plots standardized residuals Zᵢ = ε̂ᵢ / σ̂ against the predictor x.

    Overlays horizontal reference lines at 0 (black), ±1 (dashed, green),
    and ±2 (dotted, orange). Under a correctly specified model with Gaussian
    noise, roughly 68% of points should fall within ±1 and 95% within ±2.
    Patterns in this plot reveal non-linearity or heteroscedasticity.

    **Parameters**

    - `x` *(list or numpy array)* — predictor values.
    - `y` *(list or numpy array)* — response values.
    - `xlabel` *(str, optional)* — x-axis label; default `"x"`.
    - `title` *(str, optional)* — plot title; default `"Standardized Residuals"`.

    **Examples**

    ```python
    x = [0.41, 0.46, 0.44, 0.47, 0.42]
    y = [1850, 2620, 2340, 2690, 2160]
    plot_residuals(x, y, xlabel="Specific gravity")
    ```
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    s_xx  = np.sum((x - x_bar) ** 2)
    s_xy  = np.sum((x - x_bar) * (y - y_bar))
    beta1 = s_xy / s_xx
    beta0 = y_bar - beta1 * x_bar

    residuals    = y - (beta0 + beta1 * x)
    sigma_hat    = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    std_residuals = residuals / sigma_hat

    plt.figure()
    plt.scatter(x, std_residuals, color='C0', zorder=3)
    plt.axhline( 0, color='black',  linewidth=1.2)
    plt.axhline( 1, color='C2', linewidth=1, linestyle='--', label='±1')
    plt.axhline(-1, color='C2', linewidth=1, linestyle='--')
    plt.axhline( 2, color='C3', linewidth=1, linestyle=':',  label='±2')
    plt.axhline(-2, color='C3', linewidth=1, linestyle=':')
    plt.xlabel(xlabel)
    plt.ylabel("Standardized residual")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# QUICK REFERENCE — run this file directly to see examples
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Fake temperature data: N(22, 0.5²)
    fake_data = rng.normal(loc=22, scale=0.5, size=253)

    print("=== Regression scatter + fit ===")
    x_reg = [0.41, 0.46, 0.44, 0.47, 0.42, 0.39, 0.41, 0.44, 0.43, 0.44]
    y_reg = [1850, 2620, 2340, 2690, 2160, 1760, 2500, 2750, 2730, 3120]
    plot_regression(x_reg, y_reg,
                    xlabel="Specific gravity", ylabel="Crushing strength (psi)",
                    title="Wood Strength vs. Specific Gravity")
    plot_regression(x_reg, y_reg, fit=False, title="Wood Strength — raw scatter")

    print("=== Residual plot ===")
    plot_residuals(x_reg, y_reg, xlabel="Specific gravity")

    print("=== Histogram ===")
    plot_histogram(fake_data, bins=25, title="Simulated Temperature")

    print("=== Running Stats ===")
    plot_running_stats(fake_data, title="Simulated Temperature Convergence")

    print("=== Log-Likelihood ===")
    sigma2 = np.var(fake_data, ddof=1)
    plot_loglikelihood(fake_data, n=10,  sigma2=sigma2)
    plot_loglikelihood(fake_data, n=100, sigma2=sigma2)
