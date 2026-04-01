"""
# Probability Toolkit — Plots

Diagram helpers for discrete and continuous random variables.
Uses matplotlib and scipy — no other dependencies.

| Function | What it draws |
|---|---|
| `plot_pmf(values, probs, title)` | PMF stem plot for a discrete RV |
| `plot_pdf(dist, x_min, x_max, title)` | PDF curve for a continuous RV |
| `plot_cdf_discrete(values, probs, title)` | Step-function CDF for a discrete RV |
| `plot_cdf_continuous(dist, x_min, x_max, title)` | Smooth CDF curve for a continuous RV |
| `plot_normal_shaded(mean, std, shade_from, shade_to, title)` | Normal curve with shaded probability region |
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# =============================================================================
# 1. PMF PLOT (discrete random variable)
# Inputs:
#   values  — list of possible values X can take, e.g. [2, 4, 5, 8, 10]
#   probs   — corresponding probabilities,        e.g. [0.1, 0.2, 0.2, 0.3, 0.2]
#   title   — string label for the plot
# =============================================================================
def plot_pmf(values, probs, title="PMF of X"):
    """
    Stem plot of the **PMF** for a discrete random variable.

    - `values` — possible values X can take, e.g. `[2, 4, 5, 8, 10]`
    - `probs`  — corresponding probabilities, e.g. `[0.1, 0.2, 0.2, 0.3, 0.2]`

    ```python
    plot_pmf([2, 4, 5, 8, 10], [0.1, 0.2, 0.2, 0.3, 0.2], title="PMF of X")
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
# Inputs:
#   dist    — a scipy.stats distribution object, e.g. stats.expon(scale=2)
#   x_min   — left bound of x-axis
#   x_max   — right bound of x-axis
#   title   — string label for the plot
# Common distributions:
#   stats.norm(loc=mean, scale=std)
#   stats.expon(scale=1/lambda)
#   stats.uniform(loc=a, scale=b-a)
# =============================================================================
def plot_pdf(dist, x_min, x_max, title="PDF of X"):
    """
    Smooth **PDF** curve for a continuous random variable.

    - `dist` — a frozen `scipy.stats` distribution
    - `x_min`, `x_max` — x-axis bounds

    Common distributions:

    ```python
    stats.norm(loc=mean, scale=std)
    stats.expon(scale=1/lam)        # note: scale = 1/lambda
    stats.uniform(loc=a, scale=b-a)
    ```

    Example:

    ```python
    plot_pdf(stats.expon(scale=2), x_min=0, x_max=10, title="Exponential PDF")
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
# Inputs:
#   values  — list of possible values X can take, e.g. [2, 4, 5, 8, 10]
#   probs   — corresponding probabilities,        e.g. [0.1, 0.2, 0.2, 0.3, 0.2]
#   title   — string label for the plot
# =============================================================================
def plot_cdf_discrete(values, probs, title="CDF of X (Discrete)"):
    """
    Step-function **CDF** for a discrete random variable.

    Draws open circles at jump boundaries and filled circles at landed values
    (right-continuous convention: F(x) = P(X ≤ x)).

    - `values` — possible values X can take, e.g. `[2, 4, 5, 8, 10]`
    - `probs`  — corresponding probabilities, e.g. `[0.1, 0.2, 0.2, 0.3, 0.2]`

    ```python
    plot_cdf_discrete([2, 4, 5, 8, 10], [0.1, 0.2, 0.2, 0.3, 0.2])
    ```
    """
    # Sort by value just in case
    sorted_pairs = sorted(zip(values, probs))
    vals, ps = zip(*sorted_pairs)

    cum_probs = np.cumsum(ps)

    # Add a point before the first value so the step starts at 0
    x_steps = [vals[0] - 1] + list(vals)
    y_steps = [0] + list(cum_probs)

    plt.figure()
    plt.step(x_steps, y_steps, where='post', color='C0', linewidth=2)
    # Open circles at jumps (left-continuous reminder)
    plt.plot(vals, [0] + list(cum_probs[:-1]), 'o',
             markerfacecolor='white', markeredgecolor='C0', zorder=5)
    # Filled circles at landed values
    plt.plot(vals, cum_probs, 'o', color='C0', zorder=5)
    plt.xlabel("x")
    plt.ylabel("F(x) = P(X ≤ x)")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. CDF PLOT — CONTINUOUS (smooth curve)
# Inputs:
#   dist    — a scipy.stats distribution object, e.g. stats.norm(loc=0, scale=1)
#   x_min   — left bound of x-axis
#   x_max   — right bound of x-axis
#   title   — string label for the plot
# =============================================================================
def plot_cdf_continuous(dist, x_min, x_max, title="CDF of X (Continuous)"):
    """
    Smooth **CDF** curve for a continuous random variable.

    - `dist` — a frozen `scipy.stats` distribution
    - `x_min`, `x_max` — x-axis bounds

    ```python
    plot_cdf_continuous(stats.norm(loc=0, scale=1), x_min=-4, x_max=4)
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
# Use for: P(a < X < b), P(X > a), P(X < b)
# Inputs:
#   mean    — mean of the normal distribution
#   std     — standard deviation
#   shade_from — left boundary of shaded region (use -np.inf for "less than" tails)
#   shade_to   — right boundary of shaded region (use  np.inf for "greater than" tails)
#   title   — string label for the plot
# Example: P(1 < X < 3) for X ~ N(2, 1)
#   plot_normal_shaded(mean=2, std=1, shade_from=1, shade_to=3)
# =============================================================================
def plot_normal_shaded(mean, std, shade_from, shade_to, title="Normal Distribution"):
    """
    Normal curve with a **shaded probability region**.

    Use for P(a < X < b), P(X > a), or P(X < b).
    Use `-np.inf` / `np.inf` for one-sided tails.

    - `mean`, `std` — distribution parameters (**std**, not variance)
    - `shade_from`, `shade_to` — boundaries of the shaded region

    ```python
    plot_normal_shaded(mean=2, std=1, shade_from=1, shade_to=3)   # P(1 < X < 3)
    plot_normal_shaded(mean=0, std=1, shade_from=1.96, shade_to=np.inf)  # P(X > 1.96)
    ```
    """
    dist = stats.norm(loc=mean, scale=std)

    # Plot range: mean ± 4 std
    x_min = mean - 4 * std
    x_max = mean + 4 * std
    x = np.linspace(x_min, x_max, 500)
    y = dist.pdf(x)

    # Clip shade boundaries to plot range
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
    # Problem 4.5 from class
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

    # Standard normal with shading: P(-1 < X < 1)
    plot_normal_shaded(mean=0, std=1, shade_from=-1, shade_to=1,
                       title="Standard Normal: P(-1 < X < 1)")

    # Exponential distribution PDF and CDF
    exp_dist = stats.expon(scale=2)   # scale = 1/lambda, so lambda = 0.5
    plot_pdf(exp_dist, x_min=0, x_max=10, title="Exponential PDF (λ = 0.5)")
    plot_cdf_continuous(exp_dist, x_min=0, x_max=10, title="Exponential CDF (λ = 0.5)")