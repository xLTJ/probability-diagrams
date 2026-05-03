"""
# Probability Toolkit — Plots

Diagram helpers for discrete and continuous random variables.

| Function | What it draws |
|---|---|
| `plot_pmf(values, probs)` | PMF stem plot for a discrete RV |
| `plot_cdf_discrete(values, probs)` | Step-function CDF for a discrete RV |
| `plot_pdf(dist, x_min, x_max)` | PDF curve for a continuous RV |
| `plot_cdf_continuous(dist, x_min, x_max)` | Smooth CDF curve for a continuous RV |
| `plot_normal_shaded(mu, sigma2, shade_from, shade_to)` | Normal curve with shaded probability region |

**Building a `dist` object** (needed for `plot_pdf` and `plot_cdf_continuous`):

```python
stats.norm(loc=mu, scale=std)        # Normal  — scale is std, NOT variance
stats.expon(scale=1/lam)             # Exponential — scale is 1/λ
stats.uniform(loc=a, scale=b-a)      # Uniform on [a, b]
```
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# =============================================================================
# 1. PMF PLOT (discrete random variable)
# =============================================================================

def plot_pmf(values, probs, title="PMF of X"):
    """
    Stem plot of P(X = x) for a discrete random variable.

    **Parameters**

    - `values` *(list of float)* — all possible values X can take, e.g. `[2, 4, 5, 8, 10]`.
    - `probs` *(list of float)* — corresponding probabilities, e.g. `[0.1, 0.2, 0.2, 0.3, 0.2]`.
      Must sum to 1 and match `values` in length.
    - `title` *(str, optional)* — plot title; default `"PMF of X"`.

    **Examples**

    ```python
    plot_pmf([2, 4, 5, 8, 10], [0.1, 0.2, 0.2, 0.3, 0.2])
    plot_pmf([0, 1, 2, 3], [0.1, 0.4, 0.4, 0.1], title="PMF of Y")
    ```
    """
    plt.figure()
    plt.stem(values, probs, markerfmt='C0o', linefmt='C0-', basefmt='k-')
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.title(title)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. PDF PLOT (continuous random variable)
# =============================================================================

def plot_pdf(dist, x_min, x_max, title="PDF of X"):
    """
    PDF curve f(x) for a continuous random variable.

    **Parameters**

    - `dist` *(scipy.stats frozen distribution)* — the distribution to plot.
      See module docstring for how to build one.
    - `x_min` *(float)* — left edge of the x-axis.
    - `x_max` *(float)* — right edge of the x-axis.
    - `title` *(str, optional)* — plot title; default `"PDF of X"`.

    > **Note:** For `stats.norm`, `scale` is the standard deviation σ, not the variance σ².
    > For `stats.expon`, `scale` is 1/λ, not λ itself.

    **Examples**

    ```python
    plot_pdf(stats.expon(scale=2), 0, 10)            # Exponential(λ = 0.5)
    plot_pdf(stats.norm(loc=5, scale=2), -1, 11)     # Normal(μ = 5, σ = 2)
    ```
    """
    x = np.linspace(x_min, x_max, 500)
    y = dist.pdf(x)

    plt.figure()
    plt.plot(x, y, color='C0', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. CDF PLOT — DISCRETE (step function)
# =============================================================================

def plot_cdf_discrete(values, probs, title="CDF of X (Discrete)"):
    """
    Step-function CDF F(x) = P(X ≤ x) for a discrete random variable.

    Open circles mark the left edge of each jump (value not yet included);
    filled circles mark the landed value (value now included).

    **Parameters**

    - `values` *(list of float)* — all possible values X can take, e.g. `[2, 4, 5, 8, 10]`.
    - `probs` *(list of float)* — corresponding probabilities, e.g. `[0.1, 0.2, 0.2, 0.3, 0.2]`.
      Must sum to 1 and match `values` in length.
    - `title` *(str, optional)* — plot title; default `"CDF of X (Discrete)"`.

    **Examples**

    ```python
    plot_cdf_discrete([2, 4, 5, 8, 10], [0.1, 0.2, 0.2, 0.3, 0.2])
    ```
    """
    sorted_pairs = sorted(zip(values, probs))
    vals, ps = zip(*sorted_pairs)

    cum_probs = np.cumsum(ps)

    x_steps = [vals[0] - 1] + list(vals)
    y_steps = [0] + list(cum_probs)

    plt.figure()
    plt.step(x_steps, y_steps, where='post', color='C0', linewidth=2)
    plt.plot(vals, [0] + list(cum_probs[:-1]), 'o',
             markerfacecolor='white', markeredgecolor='C0', zorder=5)
    plt.plot(vals, cum_probs, 'o', color='C0', zorder=5)
    plt.xlabel("x")
    plt.ylabel("F(x) = P(X ≤ x)")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. CDF PLOT — CONTINUOUS (smooth curve)
# =============================================================================

def plot_cdf_continuous(dist, x_min, x_max, title="CDF of X (Continuous)"):
    """
    Smooth CDF curve F(x) = P(X ≤ x) for a continuous random variable.

    **Parameters**

    - `dist` *(scipy.stats frozen distribution)* — the distribution to plot.
      See module docstring for how to build one.
    - `x_min` *(float)* — left edge of the x-axis.
    - `x_max` *(float)* — right edge of the x-axis.
    - `title` *(str, optional)* — plot title; default `"CDF of X (Continuous)"`.

    > **Note:** For `stats.norm`, `scale` is the standard deviation σ, not the variance σ².
    > For `stats.expon`, `scale` is 1/λ, not λ itself.

    **Examples**

    ```python
    plot_cdf_continuous(stats.norm(loc=0, scale=1), -4, 4)   # Standard normal
    plot_cdf_continuous(stats.expon(scale=2), 0, 10)         # Exponential(λ = 0.5)
    ```
    """
    x = np.linspace(x_min, x_max, 500)
    y = dist.cdf(x)

    plt.figure()
    plt.plot(x, y, color='C0', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("F(x) = P(X ≤ x)")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. NORMAL CURVE WITH SHADED REGION
# =============================================================================

def plot_normal_shaded(mu, sigma2, shade_from, shade_to, title="Normal Distribution"):
    """
    Normal curve with a shaded probability region for X ~ N(μ, σ²).

    Useful for visualising P(a < X < b), P(X > a), or P(X < b).
    The shaded area label shows the exact probability.

    **Parameters**

    - `mu` *(float)* — mean μ.
    - `sigma2` *(float)* — variance σ² — **NOT standard deviation**. Pass `36` to get σ = 6.
    - `shade_from` *(float)* — left boundary of the shaded region. Use `-np.inf` for P(X < b).
    - `shade_to` *(float)* — right boundary of the shaded region. Use `np.inf` for P(X > a).
    - `title` *(str, optional)* — plot title; default `"Normal Distribution"`.

    **Examples**

    ```python
    plot_normal_shaded(2, 1, 1, 3)                    # P(1 < X < 3),  X ~ N(2, 1)
    plot_normal_shaded(0, 1, 1.96, np.inf)            # P(X > 1.96),   X ~ N(0, 1)
    plot_normal_shaded(10, 36, -np.inf, 5)            # P(X < 5),      X ~ N(10, 36)
    ```
    """
    std = np.sqrt(sigma2)
    dist = stats.norm(loc=mu, scale=std)

    x_min = mu - 4 * std
    x_max = mu + 4 * std
    x = np.linspace(x_min, x_max, 500)
    y = dist.pdf(x)

    shade_from_clipped = max(shade_from, x_min)
    shade_to_clipped   = min(shade_to,   x_max)

    x_shade = np.linspace(shade_from_clipped, shade_to_clipped, 500)
    y_shade = dist.pdf(x_shade)

    prob = dist.cdf(shade_to) - dist.cdf(shade_from)

    plt.figure()
    plt.plot(x, y, color='C0', linewidth=2)
    plt.fill_between(x_shade, y_shade, alpha=0.4, color='C0',
                     label=f"P = {prob:.4f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    plot_pmf(
        values=[2, 4, 5, 8, 10],
        probs=[0.1, 0.2, 0.2, 0.3, 0.2],
        title="PMF of X (Problem 4.5)"
    )

    plot_cdf_discrete(
        values=[2, 4, 5, 8, 10],
        probs=[0.1, 0.2, 0.2, 0.3, 0.2],
        title="CDF of X (Problem 4.5)"
    )

    plot_normal_shaded(mu=0, sigma2=1, shade_from=-1, shade_to=1,
                       title="Standard Normal: P(-1 < X < 1)")

    exp_dist = stats.expon(scale=2)   # scale = 1/lambda, so lambda = 0.5
    plot_pdf(exp_dist, x_min=0, x_max=10, title="Exponential PDF (λ = 0.5)")
    plot_cdf_continuous(exp_dist, x_min=0, x_max=10, title="Exponential CDF (λ = 0.5)")
