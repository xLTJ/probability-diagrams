"""
# Probability Toolkit — Compute

Numerical calculations for common distributions.
Every function prints a labelled result **and** returns the value.

| Function | What it computes |
|---|---|
| `normal_prob(mu, sigma2, a, b)` | P(a < X < b) for X ~ N(mu, sigma²) |
| `poisson_prob(lam, a, b)` | P(a ≤ X ≤ b) for X ~ Poisson(λ) |
| `binom_prob(n, p, a, b)` | P(a ≤ X ≤ b) for X ~ Binomial(n, p) |
| `geom_prob(p, k, a, b)` | P(X = k) or range for X ~ Geometric(p) |
| `phi(z)` | Standard normal CDF: Φ(z) |
| `standardize(x, mu, sigma2)` | Convert x to Z-score |
"""

import numpy as np
from scipy import stats


# =============================================================================
# STANDARD NORMAL
# =============================================================================

def phi(z):
    """
    Standard normal CDF: **Φ(z) = P(Z ≤ z)** for Z ~ N(0, 1).

    ```python
    phi(1.96)   # → 0.975002
    phi(-1.96)  # → 0.024998
    ```
    """
    result = stats.norm.cdf(z)
    print(f"Phi({z}) = {result:.6f}")
    return result


def standardize(x, mu, sigma2):
    """
    Convert x to a Z-score for X ~ N(mu, sigma²): **z = (x − μ) / σ**.

    ```python
    standardize(8, mu=10, sigma2=36)  # → z = -0.3333
    ```
    """
    sigma = np.sqrt(sigma2)
    z = (x - mu) / sigma
    print(f"z = ({x} - {mu}) / {sigma:.4f} = {z:.6f}")
    return z


# =============================================================================
# NORMAL DISTRIBUTION  X ~ N(mu, sigma2)
# =============================================================================

def normal_prob(mu, sigma2, a=-np.inf, b=np.inf):
    """
    **P(a < X < b)** for X ~ N(mu, sigma²). Omit `a` or `b` for one-sided tails.

    ```python
    normal_prob(10, 36, b=5)    # P(X < 5)
    normal_prob(10, 36, a=5)    # P(X > 5)
    normal_prob(10, 36, 4, 16)  # P(4 < X < 16)
    ```
    """
    sigma = np.sqrt(sigma2)
    dist = stats.norm(loc=mu, scale=sigma)
    result = dist.cdf(b) - dist.cdf(a)

    a_str = "-inf" if a == -np.inf else str(a)
    b_str = "+inf" if b == np.inf  else str(b)
    print(f"P({a_str} < X < {b_str}) for X ~ N({mu}, {sigma2}) = {result:.6f}")
    return result


# =============================================================================
# POISSON DISTRIBUTION  X ~ Poisson(lam)
# =============================================================================

def poisson_prob(lam, a=None, b=None):
    """
    Probability for X ~ Poisson(λ). Bounds are **inclusive**.

    ```python
    poisson_prob(3, b=5)      # P(X ≤ 5)       — upper CDF
    poisson_prob(3, a=2)      # P(X ≥ 2)       — upper tail
    poisson_prob(3, a=2, b=5) # P(2 ≤ X ≤ 5)  — range
    ```
    """
    dist = stats.poisson(lam)

    if a is None and b is not None:
        result = dist.cdf(b)
        print(f"P(X <= {b}) for X ~ Poisson({lam}) = {result:.6f}")
    elif b is None and a is not None:
        result = 1 - dist.cdf(a - 1)
        print(f"P(X >= {a}) for X ~ Poisson({lam}) = {result:.6f}")
    elif a is not None and b is not None:
        result = dist.cdf(b) - dist.cdf(a - 1)
        print(f"P({a} <= X <= {b}) for X ~ Poisson({lam}) = {result:.6f}")
    else:
        raise ValueError("Provide at least one of a or b.")

    return result


# =============================================================================
# BINOMIAL DISTRIBUTION  X ~ Binomial(n, p)
# =============================================================================

def binom_prob(n, p, a=None, b=None):
    """
    Probability for X ~ Binomial(n, p). Bounds are **inclusive**.

    ```python
    binom_prob(10, 0.5, b=3)       # P(X ≤ 3)       — upper CDF
    binom_prob(10, 0.5, a=4)       # P(X ≥ 4)       — upper tail
    binom_prob(10, 0.5, a=2, b=6)  # P(2 ≤ X ≤ 6)  — range
    ```
    """
    dist = stats.binom(n, p)

    if a is None and b is not None:
        result = dist.cdf(b)
        print(f"P(X <= {b}) for X ~ Bin({n}, {p}) = {result:.6f}")
    elif b is None and a is not None:
        result = 1 - dist.cdf(a - 1)
        print(f"P(X >= {a}) for X ~ Bin({n}, {p}) = {result:.6f}")
    elif a is not None and b is not None:
        result = dist.cdf(b) - dist.cdf(a - 1)
        print(f"P({a} <= X <= {b}) for X ~ Bin({n}, {p}) = {result:.6f}")
    else:
        raise ValueError("Provide at least one of a or b.")

    return result


# =============================================================================
# GEOMETRIC DISTRIBUTION  X ~ Geometric(p)
# (X = number of trials until first success, so X >= 1)
# =============================================================================

def geom_prob(p, k=None, a=None, b=None):
    """
    Probability for X ~ Geometric(p), where **X = trial of first success** (X ≥ 1).

    P(X = k) = (1 − p)^(k−1) · p

    ```python
    geom_prob(0.3, k=4)       # P(X = 4)       — point probability
    geom_prob(0.3, b=4)       # P(X ≤ 4)       — CDF
    geom_prob(0.3, a=3)       # P(X ≥ 3)       — upper tail
    geom_prob(0.3, a=2, b=5)  # P(2 ≤ X ≤ 5)  — range
    ```
    """
    dist = stats.geom(p)

    if k is not None:
        result = dist.pmf(k)
        print(f"P(X = {k}) for X ~ Geom({p}) = {result:.6f}")
    elif a is None and b is not None:
        result = dist.cdf(b)
        print(f"P(X <= {b}) for X ~ Geom({p}) = {result:.6f}")
    elif b is None and a is not None:
        result = 1 - dist.cdf(a - 1)
        print(f"P(X >= {a}) for X ~ Geom({p}) = {result:.6f}")
    elif a is not None and b is not None:
        result = dist.cdf(b) - dist.cdf(a - 1)
        print(f"P({a} <= X <= {b}) for X ~ Geom({p}) = {result:.6f}")
    else:
        raise ValueError("Provide k for a point probability, or a/b for range.")

    return result


# =============================================================================
# QUICK REFERENCE — run this file directly to see examples
# =============================================================================
if __name__ == "__main__":
    print("=== Normal ===")
    normal_prob(10, 36, a=5)          # P(X > 5)
    normal_prob(10, 36, 4, 16)        # P(4 < X < 16)
    normal_prob(10, 36, b=8)          # P(X < 8)

    print("\n=== Phi / Z-score ===")
    phi(-0.8333)
    standardize(8, mu=10, sigma2=36)

    print("\n=== Poisson ===")
    poisson_prob(3, b=5)              # P(X <= 5)
    poisson_prob(3, a=2, b=5)         # P(2 <= X <= 5)

    print("\n=== Binomial ===")
    binom_prob(10, 0.5, b=3)          # P(X <= 3)
    binom_prob(10, 0.5, a=2, b=6)     # P(2 <= X <= 6)

    print("\n=== Geometric ===")
    geom_prob(0.3, k=4)               # P(X = 4)
    geom_prob(0.3, a=3)               # P(X >= 3)