"""
# Probability Toolkit — Compute

Numerical calculations for common distributions.
Every function prints a labelled result and returns the value.

> **Note:** All normal/gaussian functions take **variance** (σ²), not standard deviation.

| Function | What it computes |
|---|---|
| `phi(z)` | Φ(z) — standard normal CDF |
| `standardize(x, mu, sigma2)` | Z-score: (x − μ) / σ |
| `normal_prob(mu, sigma2, a, b)` | P(a < X < b) for X ~ N(μ, σ²) |
| `poisson_prob(lam, a, b)` | P(a ≤ X ≤ b) for X ~ Poisson(λ) |
| `binom_prob(n, p, a, b)` | P(a ≤ X ≤ b) for X ~ Bin(n, p) |
| `geom_prob(p, k, a, b)` | P(X = k) or range for X ~ Geom(p) |
"""

import numpy as np
from scipy import stats


# =============================================================================
# STANDARD NORMAL
# =============================================================================

def phi(z):
    """
    Φ(z) = P(Z ≤ z) for the standard normal Z ~ N(0, 1).

    **Parameters**

    - `z` *(float)* — z-score; any real number.

    **Prints**

    `Phi(<z>) = <result>`

    **Examples**

    ```python
    phi(1.96)    # → Phi(1.96) = 0.975002
    phi(-1.96)   # → Phi(-1.96) = 0.024998
    ```
    """
    result = stats.norm.cdf(z)
    print(f"Phi({z}) = {result:.6f}")
    return result


def standardize(x, mu, sigma2):
    """
    Converts an observed value x to a z-score: z = (x − μ) / σ.

    Used to express how many standard deviations x lies above or below the mean.
    Useful before calling `phi()` when working with a non-standard normal.

    **Parameters**

    - `x` *(float)* — observed value to convert.
    - `mu` *(float)* — mean μ of the distribution.
    - `sigma2` *(float)* — variance σ² — **NOT standard deviation**. Pass `36` to get σ = 6.

    **Prints**

    `z = (<x> - <mu>) / <sigma> = <result>`

    **Examples**

    ```python
    standardize(8, mu=10, sigma2=36)   # → z = (8 - 10) / 6.0 = -0.333333
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
    P(a < X < b) for X ~ N(μ, σ²).

    Provide one or both bounds. Omitting `a` gives a left-tail probability;
    omitting `b` gives a right-tail probability.

    **Parameters**

    - `mu` *(float)* — mean μ.
    - `sigma2` *(float)* — variance σ² — **NOT standard deviation**. Pass `36` to get σ = 6.
    - `a` *(float, optional)* — lower bound; default −∞. Omit for P(X < b).
    - `b` *(float, optional)* — upper bound; default +∞. Omit for P(X > a).

    > **Note:** For continuous distributions, < and ≤ give identical results.

    **Prints**

    `P(<a> < X < <b>) for X ~ N(<mu>, <sigma2>) = <result>`

    **Examples**

    ```python
    normal_prob(10, 36, b=5)     # P(X < 5)
    normal_prob(10, 36, a=5)     # P(X > 5)
    normal_prob(10, 36, 4, 16)   # P(4 < X < 16)
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
    P(a ≤ X ≤ b) for X ~ Poisson(λ).

    Provide one or both bounds. Omitting `a` gives P(X ≤ b);
    omitting `b` gives P(X ≥ a).

    **Parameters**

    - `lam` *(float)* — rate λ; the expected number of events in the interval.
    - `a` *(int, optional)* — lower bound (inclusive). Omit for P(X ≤ b).
    - `b` *(int, optional)* — upper bound (inclusive). Omit for P(X ≥ a).

    **Prints**

    `P(<a> <= X <= <b>) for X ~ Poisson(<lam>) = <result>`

    **Examples**

    ```python
    poisson_prob(3, b=5)       # P(X ≤ 5)
    poisson_prob(3, a=2)       # P(X ≥ 2)
    poisson_prob(3, a=2, b=5)  # P(2 ≤ X ≤ 5)
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
    P(a ≤ X ≤ b) for X ~ Bin(n, p).

    Provide one or both bounds. Omitting `a` gives P(X ≤ b);
    omitting `b` gives P(X ≥ a).

    **Parameters**

    - `n` *(int)* — number of trials.
    - `p` *(float)* — probability of success on each trial; must be in [0, 1].
    - `a` *(int, optional)* — lower bound (inclusive). Omit for P(X ≤ b).
    - `b` *(int, optional)* — upper bound (inclusive). Omit for P(X ≥ a).

    **Prints**

    `P(<a> <= X <= <b>) for X ~ Bin(<n>, <p>) = <result>`

    **Examples**

    ```python
    binom_prob(10, 0.5, b=3)       # P(X ≤ 3)
    binom_prob(10, 0.5, a=4)       # P(X ≥ 4)
    binom_prob(10, 0.5, a=2, b=6)  # P(2 ≤ X ≤ 6)
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
# =============================================================================

def geom_prob(p, k=None, a=None, b=None):
    """
    Point probability or range probability for X ~ Geom(p).

    > **Convention:** X counts the trial number of the **first success**, so X ≥ 1.
    > This is the standard textbook convention. The alternative (counting failures
    > before the first success, X ≥ 0) is **not** used here.

    Use `k` for a single point probability P(X = k), or `a`/`b` for a range.
    Do not pass both `k` and `a`/`b` — if `k` is provided it takes priority.

    **Parameters**

    - `p` *(float)* — probability of success on each trial; must be in (0, 1].
    - `k` *(int, optional)* — trial number for a point probability P(X = k).
    - `a` *(int, optional)* — lower bound (inclusive) for a range probability.
    - `b` *(int, optional)* — upper bound (inclusive) for a range probability.

    **Prints**

    `P(X = <k>) for X ~ Geom(<p>) = <result>` — for point probabilities

    `P(<a> <= X <= <b>) for X ~ Geom(<p>) = <result>` — for ranges

    **Examples**

    ```python
    geom_prob(0.3, k=4)       # P(X = 4)
    geom_prob(0.3, b=4)       # P(X ≤ 4)
    geom_prob(0.3, a=3)       # P(X ≥ 3)
    geom_prob(0.3, a=2, b=5)  # P(2 ≤ X ≤ 5)
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
        raise ValueError("Provide k for a point probability, or a/b for a range.")

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
