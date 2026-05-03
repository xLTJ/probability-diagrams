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
| `load_csv(filepath, col, header)` | Load a single CSV column as a numpy array |
| `sample_stats(data)` | Sample mean, variance, std, and n from a dataset |
| `z_quantile(alpha, side)` | Z_{α/2} or Z_{α} — normal quantile; `side='two'` (default) for CIs, `'one'` for one-sided tests |
| `t_quantile(alpha, n, side, df)` | t quantile; `side='two'`/`'one'`; pass `df` directly for regression (df = n−2) |
| `pvalue(T, side, df)` | p-value from a hand-computed test statistic; pass `df` for t-tests, omit for z-tests |
| `conf_interval(data, confidence, sigma2)` | Confidence interval for μ — auto-selects Z or t method |
| `regression_fit(x, y)` | Least squares fit: prints s_xx, s_xy, β̂₀, β̂₁, σ̂², R², residuals, and sum-to-zero check |
| `regression_slope_test(x, y, beta_H, alpha, side)` | Slope hypothesis test: prints se(β̂₁), T, t-critical, p-value, and decision |
| `regression_predict(x, y, x_star, alpha)` | CI for mean response and prediction interval at a new point x* |
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
    df = pd.read_csv(filepath, header=0 if header else None)
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
# CONFIDENCE INTERVAL QUANTILES
# =============================================================================

def z_quantile(alpha, side='two'):
    """
    Standard normal quantile for CI construction or hypothesis testing.

    - **Two-sided** (`side='two'`): returns Z_{α/2}, i.e. Φ⁻¹(1 − α/2).
      Use for two-sided CIs and two-sided hypothesis tests.
    - **One-sided** (`side='one'`): returns Z_α, i.e. Φ⁻¹(1 − α).
      Use for one-sided hypothesis tests (upper- or lower-tail).

    **Parameters**

    - `alpha` *(float)* — significance level α; use 0.05 for a 95% CI or a
      5% significance level test.
    - `side` *(str, optional)* — `'two'` (default) or `'one'`.

    **Prints**

    `Z_{α/2} = <r>` or `Z_{α} = <r>` depending on `side`.

    **Examples**

    ```python
    z_quantile(0.05)             # → Z_{0.025} = 1.959964  (two-sided, 95% CI)
    z_quantile(0.05, side='one') # → Z_{0.05}  = 1.644854  (one-sided test)
    z_quantile(0.01, side='one') # → Z_{0.01}  = 2.326348  (one-sided test)
    ```
    """
    if side == 'one':
        z = stats.norm.ppf(1 - alpha)
        print(f"Z_{{α}} = {z:.6f}   (one-sided, α = {alpha})")
    else:
        z = stats.norm.ppf(1 - alpha / 2)
        print(f"Z_{{α/2}} = {z:.6f}   (two-sided, α = {alpha})")
    return z
    return z


def t_quantile(alpha, n=None, side='two', df=None):
    """
    t-distribution quantile for CI construction or hypothesis testing.

    - **Two-sided** (`side='two'`): returns T_{α/2, df}, i.e. the value t
      such that P(T > t) = α/2. Use for two-sided CIs and tests.
    - **One-sided** (`side='one'`): returns T_{α, df}, i.e. the value t
      such that P(T > t) = α. Use for one-sided hypothesis tests.

    Degrees of freedom are set to n − 1 by default (CI and z-test context).
    Pass `df` explicitly when a different value is needed — e.g. df = n − 2
    for regression slope tests.

    **Parameters**

    - `alpha` *(float)* — significance level α; use 0.05 for a 95% CI.
    - `n` *(int, optional)* — sample size; sets df = n − 1. Ignored if `df` is provided.
    - `side` *(str, optional)* — `'two'` (default) or `'one'`.
    - `df` *(int, optional)* — degrees of freedom. Overrides `n` when provided.
      Use `df=n-2` for regression slope tests.

    **Prints**

    `T_{α/2, df} = <r>` or `T_{α, df} = <r>` depending on `side`.

    **Examples**

    ```python
    t_quantile(0.05, n=9)              # → T_{0.025, 8} = 2.306004  (95% CI, n=9)
    t_quantile(0.05, n=9, side='one')  # → T_{0.05,  8} = 1.859548  (one-sided)
    t_quantile(0.05, df=10)            # → T_{0.025, 10}            (regression, n=12)
    t_quantile(0.05, df=5, side='two') # → T_{0.025,  5}            (regression, n=7)
    ```
    """
    if df is None:
        if n is None:
            raise ValueError("Provide either n or df.")
        df = n - 1

    if side == 'one':
        t = stats.t.ppf(1 - alpha, df=df)
        print(f"T_{{α, df}} = {t:.6f}   (df = {df}, one-sided, α = {alpha})")
    else:
        t = stats.t.ppf(1 - alpha / 2, df=df)
        print(f"T_{{α/2, df}} = {t:.6f}   (df = {df}, two-sided, α = {alpha})")
    return t


# =============================================================================
# CONFIDENCE INTERVAL
# =============================================================================

def conf_interval(data, confidence=0.95, sigma2=None):
    """
    Confidence interval for the population mean μ.

    Automatically selects the appropriate method based on what is known:

    - **σ² known** (`sigma2` provided): uses Z_{α/2} with the true σ.
      Exact when data are normal; approximate via CLT for large n.
    - **σ² unknown** (`sigma2=None`): estimates variance from data using S²,
      then uses T_{α/2, n−1}. Valid for any n when data are approximately normal;
      for large n this converges to the Z result.

    > **Note:** For large n with unknown σ², the t-CI and Z-CI (with S_n) give
    > nearly identical results. The t-CI is always the safer choice.

    **Parameters**

    - `data` *(list or numpy array)* — the raw observations.
    - `confidence` *(float, optional)* — confidence level 1 − α; default `0.95`.
    - `sigma2` *(float or None, optional)* — known population variance σ².
      Pass `None` (default) if σ² is unknown and should be estimated from data.

    **Prints**

    ```
    Method   : Z  (known σ²)   or   t  (unknown σ², df = n−1)
    n        : <sample size>
    x̄        : <sample mean>
    σ or S   : <std used>
    quantile : <Z or T value>
    CI       : (<lower>, <upper>)
    half-width: <half-width>
    ```

    **Examples**

    ```python
    # Known variance
    data = [5, 8.5, 12, 15, 7, 9, 7.5, 6.5, 10.5]
    conf_interval(data, confidence=0.95, sigma2=4)

    # Unknown variance — t method
    conf_interval(data, confidence=0.95)

    # Load from CSV first
    data = load_csv("temperature.csv", col=1, header=False)
    conf_interval(data, confidence=0.99)
    ```
    """
    data  = np.asarray(data, dtype=float)
    n     = len(data)
    x_bar = np.mean(data)
    alpha = 1 - confidence

    if sigma2 is not None:
        # Known variance — Z method
        sigma    = np.sqrt(sigma2)
        quantile = stats.norm.ppf(1 - alpha / 2)
        method   = f"Z  (known σ² = {sigma2})"
        std_used = sigma
    else:
        # Unknown variance — t method
        s2       = np.var(data, ddof=1)
        std_used = np.sqrt(s2)
        quantile = stats.t.ppf(1 - alpha / 2, df=n - 1)
        method   = f"t  (unknown σ², df = {n-1})"

    half_width = quantile * std_used / np.sqrt(n)
    lower      = x_bar - half_width
    upper      = x_bar + half_width

    label = "σ" if sigma2 is not None else "S"
    print(f"Method    : {method}")
    print(f"n         : {n}")
    print(f"x̄         : {x_bar:.6f}")
    print(f"{label}         : {std_used:.6f}")
    print(f"quantile  : {quantile:.6f}")
    print(f"CI        : ({lower:.6f}, {upper:.6f})")
    print(f"half-width: {half_width:.6f}")

    return lower, upper


# =============================================================================
# REGRESSION
# =============================================================================

def regression_fit(x, y):
    """
    Least squares fit for the simple linear regression model Y = β₀ + β₁X + ε.

    Computes and prints all intermediate quantities you need to verify your
    hand calculations, then the final estimates and diagnostics.

    **Printed output**

    ```
    n        = <number of pairs>
    x̄        = <mean of x>
    ȳ        = <mean of y>
    s_xx     = Σ(xᵢ − x̄)²
    s_xy     = Σ(xᵢ − x̄)(yᵢ − ȳ)
    s_yy     = Σ(yᵢ − ȳ)²
    β̂₁       = s_xy / s_xx
    β̂₀       = ȳ − β̂₁ · x̄
    σ̂²       = RSS / (n − 2)
    R²       = s_xy² / (s_xx · s_yy)
    residuals: [ε̂₁, ε̂₂, ...]
    Σε̂ᵢ     = <should be ~0>
    ```

    **Parameters**

    - `x` *(list or numpy array)* — predictor values.
    - `y` *(list or numpy array)* — response values; must be the same length as `x`.

    **Returns**

    `(beta0, beta1, sigma2)` — the three fitted quantities most likely needed
    for subsequent steps (slope test, prediction).

    **Examples**

    ```python
    x = [0.41, 0.46, 0.44, 0.47, 0.42]
    y = [1850, 2620, 2340, 2690, 2160]
    beta0, beta1, sigma2 = regression_fit(x, y)
    ```
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    s_xx  = np.sum((x - x_bar) ** 2)
    s_xy  = np.sum((x - x_bar) * (y - y_bar))
    s_yy  = np.sum((y - y_bar) ** 2)

    beta1 = s_xy / s_xx
    beta0 = y_bar - beta1 * x_bar

    residuals = y - (beta0 + beta1 * x)
    rss       = np.sum(residuals ** 2)
    sigma2    = rss / (n - 2)
    r2        = s_xy ** 2 / (s_xx * s_yy)

    print(f"n         = {n}")
    print(f"x̄         = {x_bar:.6f}")
    print(f"ȳ         = {y_bar:.6f}")
    print(f"s_xx      = {s_xx:.6f}")
    print(f"s_xy      = {s_xy:.6f}")
    print(f"s_yy      = {s_yy:.6f}")
    print(f"β̂₁        = {beta1:.6f}")
    print(f"β̂₀        = {beta0:.6f}")
    print(f"σ̂²        = {sigma2:.6f}  (RSS / (n-2))")
    print(f"R²        = {r2:.6f}")
    print(f"residuals : {np.round(residuals, 4).tolist()}")
    print(f"Σε̂ᵢ       = {np.sum(residuals):.2e}  (should be ~0)")

    return beta0, beta1, sigma2


def regression_slope_test(x, y, beta_H=0, alpha=0.05, side='two'):
    """
    Hypothesis test for the slope β₁ in simple linear regression.

    Tests H₀: β₁ = β_H (default 0) against H₁ determined by `side`.
    Prints every intermediate quantity from the standard procedure so you
    can verify each step of your hand calculation:

    ```
    β̂₁        = <slope>
    σ̂²        = RSS / (n-2)
    S_X       = sqrt(s_xx / n)
    se(β̂₁)    = σ̂ / (S_X · √n)
    T         = |β̂₁ − β_H| / se(β̂₁)
    t_{α,df}  = <critical value>   (df = n-2)
    p-value   = <p>
    Decision  : Reject H₀  /  Fail to reject H₀
    ```

    **Parameters**

    - `x` *(list or numpy array)* — predictor values.
    - `y` *(list or numpy array)* — response values.
    - `beta_H` *(float, optional)* — hypothesised slope value; default `0`.
    - `alpha` *(float, optional)* — significance level; default `0.05`.
    - `side` *(str, optional)* — `'two'` (default), `'upper'`, or `'lower'`.

    **Examples**

    ```python
    # Ex 2 / fuel consumption example — test whether slope is zero
    x = [45, 50, 55, 60, 65, 70, 75]
    y = [24.2, 25.0, 23.3, 22.0, 21.5, 20.6, 19.8]
    regression_slope_test(x, y)                        # H₀: β₁ = 0, two-sided

    # Test against a specific non-zero slope
    regression_slope_test(x, y, beta_H=0.5, alpha=0.01, side='upper')
    ```
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    df = n - 2

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    s_xx  = np.sum((x - x_bar) ** 2)
    s_xy  = np.sum((x - x_bar) * (y - y_bar))
    beta1 = s_xy / s_xx

    residuals = y - ((y_bar - beta1 * x_bar) + beta1 * x)
    sigma2    = np.sum(residuals ** 2) / df
    sigma_hat = np.sqrt(sigma2)

    S_X    = np.sqrt(s_xx / n)
    se     = sigma_hat / (S_X * np.sqrt(n))
    T      = (beta1 - beta_H) / se          # signed — needed for one-sided tests

    if side == 'two':
        t_crit = stats.t.ppf(1 - alpha / 2, df=df)
        p      = 2 * (1 - stats.t.cdf(abs(T), df=df))
        reject = abs(T) > t_crit
    elif side == 'upper':
        t_crit = stats.t.ppf(1 - alpha, df=df)
        p      = 1 - stats.t.cdf(T, df=df)
        reject = T > t_crit
    elif side == 'lower':
        t_crit = stats.t.ppf(alpha, df=df)
        p      = stats.t.cdf(T, df=df)
        reject = T < t_crit
    else:
        raise ValueError("side must be 'two', 'upper', or 'lower'.")

    decision = "Reject H₀" if reject else "Fail to reject H₀"

    print(f"β̂₁        = {beta1:.6f}")
    print(f"σ̂²        = {sigma2:.6f}  (RSS / (n-2))")
    print(f"S_X       = {S_X:.6f}  (sqrt(s_xx / n))")
    print(f"se(β̂₁)    = {se:.6f}  (σ̂ / (S_X · √n))")
    print(f"T         = {T:.6f}  ((β̂₁ − {beta_H}) / se)")
    print(f"t_crit    = {t_crit:.6f}  (df = {df}, α = {alpha}, {side}-sided)")
    print(f"p-value   = {p:.6f}  (t, df = {df})")
    print(f"Decision  : {decision}  (α = {alpha})")

    return T, p


def regression_predict(x, y, x_star, alpha=0.05):
    """
    Confidence interval for the mean response and prediction interval at a
    new predictor value x*, for the simple linear regression model.

    Both intervals are centred on ŷ* = β̂₀ + β̂₁ · x*. The only difference
    is the extra leading 1 inside the square root of the prediction interval,
    which accounts for the irreducible noise in a single new observation.

    **Printed output**

    ```
    x*         = <x_star>
    ŷ*         = β̂₀ + β̂₁ · x*
    t_{α/2}    = <t-critical value, df = n-2>
    CI (mean)  : (<lower>, <upper>)   ← for E[Y | X = x*]
    PI (new)   : (<lower>, <upper>)   ← for a single new Y* at x*
    ```

    **Parameters**

    - `x` *(list or numpy array)* — predictor values used to fit the model.
    - `y` *(list or numpy array)* — response values used to fit the model.
    - `x_star` *(float)* — new predictor value at which to predict.
    - `alpha` *(float, optional)* — significance level; default 0.05 (95% intervals).

    **Examples**

    ```python
    x = [0.41, 0.46, 0.44, 0.47, 0.42, 0.39, 0.41, 0.44, 0.43, 0.44]
    y = [1850, 2620, 2340, 2690, 2160, 1760, 2500, 2750, 2730, 3120]
    regression_predict(x, y, x_star=0.43)           # 95% PI and CI
    regression_predict(x, y, x_star=3400, alpha=0.10)  # 90% CI
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

    residuals = y - (beta0 + beta1 * x)
    sigma2    = np.sum(residuals ** 2) / (n - 2)

    y_star  = beta0 + beta1 * x_star
    t_crit  = stats.t.ppf(1 - alpha / 2, df=n - 2)

    se_mean = np.sqrt(sigma2 * (1/n + (x_star - x_bar) ** 2 / s_xx))
    se_pred = np.sqrt(sigma2 * (1 + 1/n + (x_star - x_bar) ** 2 / s_xx))

    ci = (y_star - t_crit * se_mean, y_star + t_crit * se_mean)
    pi = (y_star - t_crit * se_pred, y_star + t_crit * se_pred)

    print(f"x*         = {x_star}")
    print(f"ŷ*         = {y_star:.6f}")
    print(f"t_{{α/2}}    = {t_crit:.6f}   (df = {n-2}, α = {alpha})")
    print(f"CI (mean)  : ({ci[0]:.6f}, {ci[1]:.6f})")
    print(f"PI (new)   : ({pi[0]:.6f}, {pi[1]:.6f})")

    return ci, pi


# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

def pvalue(T, side, df=None):
    """
    p-value from an already-computed test statistic under H₀.

    Defaults to the standard normal N(0,1), which is correct for z-tests
    (known σ², large-sample proportion tests). Pass `df` to use the
    t-distribution instead — required for regression slope tests (df = n−2)
    and t-tests with unknown σ² (df = n−1).

    | `side`    | Formula              | Use when                       |
    |-----------|----------------------|--------------------------------|
    | `'two'`   | 2·(1 − CDF(|T|))     | H₁: θ ≠ θ₀                    |
    | `'upper'` | 1 − CDF(T)           | H₁: θ > θ₀  (upper-tail test) |
    | `'lower'` | CDF(T)               | H₁: θ < θ₀  (lower-tail test) |

    **Parameters**

    - `T` *(float)* — test statistic value you computed by hand.
    - `side` *(str)* — `'two'`, `'upper'`, or `'lower'`.
    - `df` *(int, optional)* — degrees of freedom. If provided, uses the
      t-distribution. Omit for z-tests (normal distribution).

    **Prints**

    `p-value (<side>-sided, <distribution>) = <v>`

    **Examples**

    ```python
    pvalue(2.14, side='upper')          # z-test, one-sided upper
    pvalue(1.96, side='two')            # z-test, two-sided
    pvalue(8.13, side='two', df=5)      # regression slope test, df = n-2 = 5
    pvalue(2.50, side='two', df=8)      # t-test with unknown σ², df = n-1 = 8
    ```
    """
    if df is not None:
        dist  = stats.t(df=df)
        label = f"t, df={df}"
    else:
        dist  = stats.norm
        label = "normal"

    if side == 'two':
        v = 2 * (1 - dist.cdf(abs(T)))
        tail = "two"
    elif side == 'upper':
        v = 1 - dist.cdf(T)
        tail = "upper"
    elif side == 'lower':
        v = dist.cdf(T)
        tail = "lower"
    else:
        raise ValueError("side must be 'two', 'upper', or 'lower'.")

    print(f"p-value ({tail}-sided, {label}) = {v:.6f}")
    return v


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

    print("\n=== Quantiles ===")
    z_quantile(0.05)                       # Z for 95% CI (two-sided)
    z_quantile(0.05, side='one')           # Z for one-sided test at α=0.05
    t_quantile(0.05, n=9)                  # T for 95% CI, n=9 (df=8)
    t_quantile(0.05, n=9, side='one')      # T for one-sided test at α=0.05, n=9
    t_quantile(0.05, df=5)                 # regression context: df=n-2=5 (n=7)
    t_quantile(0.05, n=100)               # Should be close to Z

    print("\n=== p-values ===")
    pvalue(2.14, side='upper')             # z-test, one-sided upper
    pvalue(2.14, side='two')              # z-test, two-sided
    pvalue(8.13, side='two', df=5)        # regression slope test, df=n-2=5

    print("\n=== Regression ===")
    x = [45, 50, 55, 60, 65, 70, 75]
    y = [24.2, 25.0, 23.3, 22.0, 21.5, 20.6, 19.8]
    regression_fit(x, y)
    print()
    regression_slope_test(x, y)
    print()
    regression_predict(x, y, x_star=55)

    data = [5, 8.5, 12, 15, 7, 9, 7.5, 6.5, 10.5]
    conf_interval(data, confidence=0.95, sigma2=4)   # known variance
    print()
    conf_interval(data, confidence=0.95)             # unknown variance, t method
