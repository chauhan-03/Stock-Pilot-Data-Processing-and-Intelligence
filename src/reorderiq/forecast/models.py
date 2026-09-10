"""Per-series forecasters, each reduced to the same contract:

    f(y) -> float        y = dense weekly history up to and including the
                             origin week; returns the 1-step-ahead forecast.

Keeping one signature lets backtest.py evaluate hand-weighted averages,
exponential smoothing, intermittent-demand methods and ARIMA under an identical
rolling-origin protocol, so differences in the results table are differences in
method rather than differences in evaluation.

The weighting scheme is what separates these. Grouped by how each one decides
how much a past week counts:

  ASSERTED weights   -- naive, mean, wma_rows, wma_dense: the analyst picks the
                        weights outright.
  FITTED weights     -- ses, holt, ets: geometric decay w_k = a(1-a)^(k-1) with
                        `a` estimated per series by minimizing in-sample SSE, so
                        the decay rate is chosen by the data, not by hand.
  SPLIT weights      -- croston, sba: smooth demand SIZE and demand INTERVAL
                        separately, which a single average cannot do. This
                        matters here because 40.5% of panel weeks are zeros.
  LEARNED weights    -- see regressors.py: lag coefficients estimated by
                        Ridge/Lasso/Poisson rather than assumed.

`wma_rows` reproduces the incumbent forecast *including its defect* -- it drops
zero-weeks and averages the last 6 surviving rows -- so the benchmark can
separate "the bug fix helped" from "the new model helped".
"""
import warnings

import numpy as np

WINDOW = 6  # matches the incumbent's window, so comparisons hold the window fixed


# ---------------------------------------------------------------- asserted weights

def naive(y: np.ndarray) -> float:
    """Last observed week. The reference point for MASE."""
    return float(y[-1]) if len(y) else 0.0


def mean(y: np.ndarray) -> float:
    """Unweighted mean of all history -- every week counts the same."""
    return float(np.mean(y)) if len(y) else 0.0


def wma_dense(y: np.ndarray, window: int = WINDOW) -> float:
    """Incumbent's linear weights w_i = i, applied to the last `window` WEEKS of
    the densified series (zero-weeks included, as they should have been)."""
    w = y[-window:]
    if len(w) == 0:
        return 0.0
    weights = np.arange(1, len(w) + 1)
    return float(np.average(w, weights=weights))


def wma_rows(y: np.ndarray, window: int = WINDOW) -> float:
    """The incumbent AS SHIPPED: strip zero-weeks, then take the last `window`
    surviving rows. Reproduces both defects -- the window is measured in rows
    rather than weeks, and zero-weeks never enter the average -- so the results
    table can attribute the improvement correctly."""
    nz = y[y > 0]
    w = nz[-window:]
    if len(w) == 0:
        return 0.0
    weights = np.arange(1, len(w) + 1)
    return float(np.average(w, weights=weights))


# ---------------------------------------------------------------- fitted weights

_ALPHA_GRID = np.arange(0.05, 1.00, 0.05)


def _ses_sse(y: np.ndarray, alpha: float) -> float:
    level = y[0]
    sse = 0.0
    for t in range(1, len(y)):
        err = y[t] - level
        sse += err * err
        level += alpha * err
    return sse


def fit_ses_alpha(y: np.ndarray) -> float:
    """Choose the smoothing constant by minimizing in-sample one-step SSE.

    `alpha` is the whole weighting scheme: week t-k carries weight
    a(1-a)^(k-1). A high alpha means recent weeks dominate; a low alpha spreads
    weight far back. Fitting it per series is the substantive difference from
    asserting [1,2,3,4,5,6]."""
    if len(y) < 3:
        return 0.5
    return float(min(_ALPHA_GRID, key=lambda a: _ses_sse(y, a)))


def ses(y: np.ndarray) -> float:
    """Simple exponential smoothing with a per-series fitted alpha."""
    if len(y) == 0:
        return 0.0
    alpha = fit_ses_alpha(y)
    level = y[0]
    for t in range(1, len(y)):
        level += alpha * (y[t] - level)
    return float(max(level, 0.0))


def holt(y: np.ndarray) -> float:
    """Holt linear trend: a level term plus a damped-in-practice trend term.

    Included mainly as a diagnostic. On this panel a trend term is dangerous --
    the relative timeline thins by survivorship, so an unguarded trend can learn
    the artifact rather than the demand. Its benchmark score is the evidence for
    or against that concern, which is why it is measured rather than assumed."""
    if len(y) < 4:
        return ses(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from statsmodels.tsa.holtwinters import Holt
            fit = Holt(np.asarray(y, dtype=float), initialization_method="estimated").fit()
            return float(max(fit.forecast(1)[0], 0.0))
        except Exception:
            return ses(y)


def ets(y: np.ndarray) -> float:
    """Additive-trend exponential smoothing via statsmodels' ETS implementation."""
    if len(y) < 6:
        return ses(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = ExponentialSmoothing(
                np.asarray(y, dtype=float), trend="add", seasonal=None,
                initialization_method="estimated",
            ).fit()
            return float(max(fit.forecast(1)[0], 0.0))
        except Exception:
            return ses(y)


def arima(y: np.ndarray, order=(1, 0, 1)) -> float:
    """ARIMA(1,0,1). Fitted per series, so it runs on the sampled arm of the
    benchmark only -- 52k series x 4 origins of maximum-likelihood fitting is not
    something to put in a nightly pipeline."""
    if len(y) < 10 or np.all(y == y[0]):
        return ses(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            fit = _ARIMA(np.asarray(y, dtype=float), order=order).fit()
            return float(max(fit.forecast(1)[0], 0.0))
        except Exception:
            return ses(y)


# ---------------------------------------------------------------- split weights

def _croston_core(y: np.ndarray, alpha: float = 0.1):
    """Smooth demand SIZE and demand INTERVAL as two separate series.

    A plain average answers "units per week" by blending 'how big when it sells'
    with 'how often it sells'. With 40.5% zero-weeks those are different
    questions, and Croston keeps them apart: z = smoothed non-zero size,
    p = smoothed gap between non-zero weeks, forecast = z / p.
    """
    nz_idx = np.flatnonzero(y > 0)
    if len(nz_idx) == 0:
        return 0.0, 1.0
    z = float(y[nz_idx[0]])
    p = float(nz_idx[0] + 1)
    for i in range(1, len(nz_idx)):
        gap = float(nz_idx[i] - nz_idx[i - 1])
        z += alpha * (y[nz_idx[i]] - z)
        p += alpha * (gap - p)
    return z, max(p, 1e-9)


_CROSTON_GRID = np.arange(0.05, 0.55, 0.05)


def _croston_sse(y: np.ndarray, alpha: float) -> float:
    """One-step SSE for a given alpha in a single incremental sweep.

    Refitting from scratch at every cut point would be O(n^2) per series, which
    across 4k series x 4 origins x 10 candidate alphas is not affordable. The
    forecast z/p is constant between arrivals, so the errors for the zero-weeks
    in each gap can be accumulated as the sweep passes them."""
    nz = np.flatnonzero(y > 0)
    if len(nz) < 2:
        return np.inf
    z = float(y[nz[0]])
    p = float(nz[0] + 1)
    sse = 0.0
    for i in range(1, len(nz)):
        f = z / max(p, 1e-9)
        seg = y[nz[i - 1] + 1: nz[i] + 1]
        sse += float(np.sum((seg - f) ** 2))
        gap = float(nz[i] - nz[i - 1])
        z += alpha * (y[nz[i]] - z)
        p += alpha * (gap - p)
    return sse


def fit_croston_alpha(y: np.ndarray) -> float:
    """Pick Croston's smoothing constant by one-step SSE rather than defaulting
    to the textbook 0.1.

    This matters more here than usual. The relative timeline decays by
    survivorship, so a slow-adapting alpha keeps chasing a demand level that has
    already fallen -- fitting alpha lets the method defend itself against that
    instead of being penalized for a hard-coded constant."""
    if np.count_nonzero(y) < 3:
        return 0.1
    return float(min(_CROSTON_GRID, key=lambda a: _croston_sse(y, a)))


def croston(y: np.ndarray, alpha: float | None = None) -> float:
    a = fit_croston_alpha(y) if alpha is None else alpha
    z, p = _croston_core(y, a)
    return float(max(z / p, 0.0))


def sba(y: np.ndarray, alpha: float | None = None) -> float:
    """Syntetos-Boylan Approximation: Croston with its (1 - alpha/2) bias
    correction. Plain Croston is known to over-forecast intermittent demand --
    the same direction as the incumbent's zero-week defect -- so the corrected
    variant is the fairer intermittent-demand benchmark."""
    a = fit_croston_alpha(y) if alpha is None else alpha
    z, p = _croston_core(y, a)
    return float(max((1.0 - a / 2.0) * z / p, 0.0))


PER_SERIES_MODELS = {
    "naive": naive,
    "mean": mean,
    "wma_rows_incumbent": wma_rows,
    "wma_dense": wma_dense,
    "ses": ses,
    "holt": holt,
    "croston": croston,
    "sba": sba,
}

# Maximum-likelihood fits, run on the sampled arm only.
SLOW_MODELS = {
    "ets": ets,
    "arima": arima,
}
