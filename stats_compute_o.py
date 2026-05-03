"""
# Statistics Toolkit — Compute

Numerical calculations for the statistics half of the course.
Includes carried-over probability functions used in statistical reasoning,
plus sample statistics for working with real datasets.

> **Note:** All normal/gaussian functions take **variance** (σ²), not standard deviation.

| Function | What it computes |
|---|---|
| `phi(z)` | Φ(z) — standard normal CDF |
| `standardize(x, mu, sigma2)` | Z-score: (x − μ) / σ |
| `normal_prob(mu, sigma2, a, b)` | P(a < X < b) for X ~ N(μ, σ²) — used for CLT approximations |
| `binom_prob(n, p, a, b)` | P(a ≤ X ≤ b) for X ~ Bin(n, p) |
| `sample_stats(data)` | Sample mean, variance, std, and n from a dataset |
"""

import numpy as np
import pandas as pd
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

    `Phi(<z>) = <r>`

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
    Useful before calling `phi()` when working with a non-standard normal,
    and when applying the CLT to convert a sample mean to a z-score.

    **Parameters**

    - `x` *(float)* — observed value to convert.
    - `mu` *(float)* — mean μ of the distribution.
    - `sigma2` *(float)* — variance σ² — **NOT standard deviation**. Pass `36` to get σ = 6.

    **Prints**

    `z = (<x> - <mu>) / <sigma> = <r>`

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

    Frequently used in statistics for CLT approximations: if X̄ is a sample mean,
    then X̄ ~ N(μ, σ²/n) approximately, and this function computes probabilities
    directly without needing to standardize first.

    Provide one or both bounds. Omitting `a` gives a left-tail probability;
    omitting `b` gives a right-tail probability.

    **Parameters**

    - `mu` *(float)* — mean μ.
    - `sigma2` *(float)* — variance σ² — **NOT standard deviation**. Pass `36` to get σ = 6.
    - `a` *(float, optional)* — lower bound; default −∞. Omit for P(X < b).
    - `b` *(float, optional)* — upper bound; default +∞. Omit for P(X > a).

    > **Note:** For continuous distributions, < and ≤ give identical results.

    **Prints**

    `P(<a> < X < <b>) for X ~ N(<mu>, <sigma2>) = <r>`

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

    `P(<a> <= X <= <b>) for X ~ Bin(<n>, <p>) = <r>`

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
# SAMPLE STATISTICS
# =============================================================================

def sample_stats(data):
    """
    Computes key sample statistics from a dataset.

    Uses **Bessel's correction** for sample variance (divides by n − 1, not n),
    which gives an unbiased estimator of the population variance.

    **Parameters**

    - `data` *(list or numpy array)* — the raw observations.

    **Prints**

    ```
    n        = <number of observations>
    mean     = <x̄>
    variance = <S²>    (divided by n-1)
    std      = <S>
    ```

    **Examples**

    ```python
    sample_stats([2.1, 2.4, 1.9, 2.0, 2.3])
    data = load_csv("temperature.csv", col=1, header=False)
    sample_stats(data)
    ```
    """
    data = np.asarray(data)
    n    = len(data)
    mean = np.mean(data)
    var  = np.var(data, ddof=1)   # ddof=1 → divide by n-1 (Bessel's correction)
    std  = np.sqrt(var)

    print(f"n        = {n}")
    print(f"mean     = {mean:.6f}")
    print(f"variance = {var:.6f}  (divided by n-1)")
    print(f"std      = {std:.6f}")

    return n, mean, var, std


# =============================================================================
# QUICK REFERENCE — run this file directly to see examples
# =============================================================================
if __name__ == "__main__":
    print("=== Normal (CLT approximation) ===")
    normal_prob(10, 36, a=5)
    normal_prob(10, 36, 4, 16)

    print("\n=== Phi / Z-score ===")
    phi(1.96)
    standardize(8, mu=10, sigma2=36)

    print("\n=== Binomial ===")
    binom_prob(10, 0.5, a=7)          # P(H_10 >= 7)
    binom_prob(100, 0.5, a=70)        # P(H_100 >= 70)

    print("\n=== Sample Stats ===")
    sample_stats([2.1, 2.4, 1.9, 2.0, 2.3])
