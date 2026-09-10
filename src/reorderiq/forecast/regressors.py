"""Global regression models: lag weights LEARNED from data, not asserted.

The incumbent forecast answers "how much does week t-k count?" by writing the
answer down: w = [1,2,3,4,5,6]. These models estimate it instead.

    y_t = b0 + sum_k b_k * y_{t-k} + g' z_t + e_t

The b_k ARE the demand weights. Unlike a hand-built weighted average they are
not forced to be positive, monotone, or to sum to 1 -- if week t-3 genuinely
carries negative information after conditioning on t-1 and t-2, the fit says so.
Ridge shrinks them (adjacent lags are badly collinear, which makes raw OLS
coefficients unstable and un-interpretable); Lasso drives useless lags to
exactly zero, which is a direct empirical test of whether a 6-week window was
the right choice.

"Global" means one model is fitted across all series at once rather than one
model per series. Each series contributes rows, and the model borrows strength
across the panel -- the reason a sparse SKU with 9 selling weeks can still get a
sensible forecast, where a per-series fit would have almost nothing to work with.
Series identity re-enters through store, department and series-level features.

COUNT DATA AND EXPOSURE. `units` is a count, so squared-error regression is the
wrong likelihood: it permits negative predictions and assumes constant variance
where counts have variance rising with the mean. PoissonRegressor and the
Poisson-loss GBM model log E[y] instead, which cannot go negative. Both carry an
exposure offset:

    E[y_t] = exposure_t * exp(x_t' b)

fitted in sklearn by regressing the RATE y/exposure with sample_weight=exposure,
the standard equivalent formulation. This is what neutralizes the survivorship
decay described in panel.py: the model learns demand per active user, which is
stable across the relative timeline, rather than demand per week, which is not.

LEAKAGE. Every feature at week t is built from weeks <= t-1, including exposure,
which is lagged even though a planner would plausibly know current headcount.
Assuming less keeps the backtest honest.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, PoissonRegressor, Ridge

MAX_LAG = 8       # lags exposed as model features
_LAG_DEPTH = 13   # lags materialized, so 13-week windows come free off the lag matrix
LAG_COLS = [f"lag_{k}" for k in range(1, MAX_LAG + 1)]

FEATURES = LAG_COLS + [
    "roll_mean_4", "roll_mean_8", "roll_mean_13",
    "roll_std_4", "roll_std_8",
    "roll_max_8", "zero_share_8", "weeks_since_sale",
    "series_mean", "series_zero_share",
    "exposure_lag1", "exposure_ratio",
    "rel_week", "store_id", "department_id",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling / intermittency features per (store, product) series.

    Every window is derived from the LAG MATRIX (weeks t-1 .. t-13) rather than
    from a groupby-rolling over 52k groups: the lags already encode the window,
    so the aggregates collapse to vectorized row-wise numpy over a few columns.
    Building them this way also makes the no-leakage property structural rather
    than something to remember -- week t's own demand is simply not among the
    inputs.
    """
    df = panel.sort_values(["store_id", "product_id", "rel_week"]).copy()
    g = df.groupby(["store_id", "product_id"], sort=False)

    for k in range(1, _LAG_DEPTH + 1):
        df[f"lag_{k}"] = g["units"].shift(k)

    L = df[[f"lag_{k}" for k in range(1, _LAG_DEPTH + 1)]].to_numpy(dtype=float)

    with np.errstate(invalid="ignore"):
        df["roll_mean_4"] = np.nanmean(L[:, :4], axis=1)
        df["roll_mean_8"] = np.nanmean(L[:, :8], axis=1)
        df["roll_mean_13"] = np.nanmean(L[:, :13], axis=1)
        df["roll_std_4"] = np.nanstd(L[:, :4], axis=1)
        df["roll_std_8"] = np.nanstd(L[:, :8], axis=1)
        df["roll_max_8"] = np.nanmax(np.where(np.isnan(L[:, :8]), -np.inf, L[:, :8]), axis=1)
    df["roll_max_8"] = df["roll_max_8"].replace(-np.inf, np.nan)

    # Intermittency. zero_share_8 is "how often did it NOT sell recently";
    # weeks_since_sale is the gap Croston smooths explicitly -- given here to the
    # regressors so they can model the same structure.
    obs8 = ~np.isnan(L[:, :8])
    with np.errstate(invalid="ignore"):
        df["zero_share_8"] = np.where(obs8.sum(1) > 0,
                                      ((L[:, :8] == 0) & obs8).sum(1) / np.maximum(obs8.sum(1), 1),
                                      np.nan)
    sold = np.where(np.isnan(L), False, L > 0)
    first_sale = np.argmax(sold, axis=1).astype(float) + 1.0
    df["weeks_since_sale"] = np.where(sold.any(axis=1), first_sale, _LAG_DEPTH + 1.0)

    # Series-level context, as expanding stats over strictly prior weeks.
    prior_n = g.cumcount().to_numpy(dtype=float)
    prior_sum = (g["units"].cumsum() - df["units"]).to_numpy(dtype=float)
    prior_zero = (g["units"].transform(lambda s: (s == 0).cumsum())
                  - (df["units"] == 0).astype(float)).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["series_mean"] = np.where(prior_n > 0, prior_sum / np.maximum(prior_n, 1), np.nan)
        df["series_zero_share"] = np.where(prior_n > 0, prior_zero / np.maximum(prior_n, 1), np.nan)

    # Exposure, lagged. exposure_ratio tells the model whether the store's active
    # base is growing or thinning, which is how it tells real demand movement
    # apart from the survivorship decay baked into the relative timeline.
    df["exposure_lag1"] = g["active_users"].shift(1)
    df["exposure_lag2"] = g["active_users"].shift(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["exposure_ratio"] = (df["exposure_lag1"] / df["exposure_lag2"].replace(0, np.nan))

    df["roll_std_4"] = df["roll_std_4"].fillna(0.0)
    df["roll_std_8"] = df["roll_std_8"].fillna(0.0)
    df["exposure_ratio"] = df["exposure_ratio"].fillna(1.0)
    return df


# Raw integer store/department ids are ordinal nonsense to a linear model
# (department 12 is not "twice" department 6), so the linear arm drops them; the
# gradient-boosted tree splits on them happily and keeps them.
LINEAR_FEATURES = [f for f in FEATURES if f not in ("store_id", "department_id")]


def make_models(seed: int = 42) -> dict:
    """The regression arm.

    Every linear model is standardized before fitting. That is not cosmetic:
    exposure_lag1 runs to ~13,000 while the lag features sit in single digits, so
    on raw scale a single L2 penalty is effectively no penalty on exposure and a
    heavy one on the lags. Standardizing makes the shrinkage comparable across
    features; the reported lag weights are converted back to raw units afterwards
    so they stay readable as "units of demand per unit of lag".

    Squared-error models (Ridge, Lasso) give the interpretable lag weights.
    Poisson models give the right likelihood for count data with an exposure
    offset. The GBM is the non-linear arm -- it can represent interactions such
    as "long gap AND thinning store base" that no additive model can.
    """
    return {
        "ridge":       dict(kind="gaussian", features=LINEAR_FEATURES, scale=True,
                            est=Ridge(alpha=1.0)),
        "lasso":       dict(kind="gaussian", features=LINEAR_FEATURES, scale=True,
                            est=Lasso(alpha=0.01, max_iter=3000, tol=1e-3, selection="random",
                                      random_state=seed)),
        "poisson_glm": dict(kind="poisson", features=LINEAR_FEATURES, scale=True,
                            est=PoissonRegressor(alpha=1e-3, max_iter=300)),
        "gbm_poisson": dict(kind="poisson", features=FEATURES, scale=False,
                            est=HistGradientBoostingRegressor(
                                loss="poisson", max_iter=400, learning_rate=0.06,
                                max_leaf_nodes=63, min_samples_leaf=40,
                                l2_regularization=1.0, early_stopping=True,
                                validation_fraction=0.1, random_state=seed)),
    }


def _xy(df: pd.DataFrame, features: list):
    X = df[features].to_numpy(dtype=float)
    y = df["units"].to_numpy(dtype=float)
    expo = df["exposure_lag1"].to_numpy(dtype=float)
    expo = np.where(np.isfinite(expo) & (expo > 0), expo, 1.0)
    return X, y, expo


def fit_predict(spec: dict, train: pd.DataFrame, test: pd.DataFrame):
    """Fit on `train`, predict `test`. Returns (predictions, raw-scale coefs).

    Poisson models are fitted on the rate y/exposure with sample_weight=exposure,
    which is the sklearn-equivalent of the offset formulation
    E[y] = exposure * exp(x'b); predictions are multiplied back to units.
    """
    feats, kind, model = spec["features"], spec["kind"], spec["est"]
    Xtr, ytr, etr = _xy(train, feats)
    Xte, _, ete = _xy(test, feats)

    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr) & np.isfinite(etr)
    Xtr, ytr, etr = Xtr[ok], ytr[ok], etr[ok]
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

    sigma = None
    if spec["scale"]:
        mu = Xtr.mean(axis=0)
        sigma = Xtr.std(axis=0)
        sigma = np.where(sigma > 0, sigma, 1.0)
        Xtr = (Xtr - mu) / sigma
        Xte = (Xte - mu) / sigma

    if kind == "poisson":
        model.fit(Xtr, ytr / etr, sample_weight=etr)
        pred = model.predict(Xte) * ete
    elif kind == "quantile":
        model.fit(Xtr, ytr)          # quantiles of units directly; see make_quantile_models
        pred = model.predict(Xte)
    else:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

    coef = None
    if hasattr(model, "coef_"):
        b = np.asarray(model.coef_, dtype=float).ravel()
        if sigma is not None:            # back to raw units, so weights stay readable
            b = b / sigma
        coef = dict(zip(feats, np.round(b, 6).tolist()))
    return np.clip(pred, 0.0, None), coef


def predict_only(spec: dict, test: pd.DataFrame) -> np.ndarray:
    """Score an ALREADY-FITTED model on new rows.

    Multi-week projection needs to run the same fitted model repeatedly as it
    walks the horizon forward; refitting at each step would be both wasteful and
    wrong, since every step would then be a different model. Mirrors the
    exposure handling in fit_predict so the recursion and the production
    forecast agree by construction.
    """
    feats, kind, model = spec["features"], spec["kind"], spec["est"]
    X = np.nan_to_num(test[feats].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    pred = model.predict(X)
    if kind == "poisson":
        expo = test["exposure_lag1"].to_numpy(dtype=float)
        pred = pred * np.where(np.isfinite(expo) & (expo > 0), expo, 1.0)
    return np.clip(pred, 0.0, None)


# ---------------------------------------------------------------- asymmetric loss

SERVICE_TAUS = (0.80, 0.90, 0.95, 0.99)


def make_quantile_models(taus=SERVICE_TAUS, seed: int = 42) -> dict:
    """Gradient-boosted QUANTILE regression -- the direct answer to "under-
    forecasting is unacceptable, over-forecasting is tolerable".

    Squared error and absolute error are both symmetric: they are minimized by
    the conditional mean and median respectively, and either sits below actual
    demand roughly half the time. Pinball loss at tau,

        L = tau * max(y - f, 0)  +  (1 - tau) * max(f - y, 0)

    charges shortfall tau/(1-tau) times more than excess -- 19x at tau=0.95 --
    and is minimized by the conditional tau-th QUANTILE of demand. Fitting it
    directly is what makes the forecast structurally high-sided rather than
    high-sided by accident, which is what the incumbent's +1.11 bias was.

    Several taus are fitted rather than one so the result is a service-level
    frontier: each tau trades excess inventory for fill rate at a known rate,
    and the business picks the point rather than inheriting whatever asymmetry
    a bug happened to produce.

    Exposure enters as a feature here rather than as an offset -- the rate
    reparameterization used for the Poisson models is a statement about the
    conditional MEAN and does not carry over to quantiles.
    """
    return {
        f"gbm_q{int(round(t * 100))}": dict(
            kind="quantile", features=FEATURES, scale=False,
            est=HistGradientBoostingRegressor(
                loss="quantile", quantile=t, max_iter=400, learning_rate=0.06,
                max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1, random_state=seed))
        for t in taus
    }


def nb_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments NB2 dispersion: Var(y) = mu + a * mu^2.

    Weekly SKU demand is overdispersed relative to Poisson -- promotions, basket
    effects and clustered shopping all inflate the variance -- so reading a
    Poisson quantile off the fitted mean would under-state the upper tail and
    therefore under-protect service.

    Estimated as a RATIO OF SUMS (the Cameron-Trivedi estimating equation),

        a = sum[(y - mu)^2 - mu] / sum[mu^2]

    rather than as the mean of per-observation ratios. That distinction is not
    cosmetic here: 40% of panel cells are zeros, the fitted mean for those is
    near zero, and dividing each residual by its own mu^2 lets a single cell with
    mu ~ 1e-6 dominate the average -- which produced an estimate of 3.1e8 before
    this was corrected. Pooling numerator and denominator first weights each cell
    by mu^2, so near-zero cells contribute in proportion to how much they can
    actually tell us about dispersion.
    """
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    ok = np.isfinite(y) & np.isfinite(mu)
    if not ok.any():
        return 0.0
    num = np.sum((y[ok] - mu[ok]) ** 2 - mu[ok])
    den = np.sum(mu[ok] ** 2)
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 50.0))


def nb_quantile(mu: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    """tau-th quantile of NB2(mu, alpha), for turning a calibrated MEAN forecast
    into a service-level forecast.

    An alternative to fitting a separate quantile model per tau: fit the mean
    once, then read any service level off the distribution. Cheaper, and it
    keeps a single coherent demand model underneath every tau -- but it is only
    as good as the NB2 assumption, which is why the benchmark scores it against
    the distribution-free gbm_q* models instead of assuming it wins.
    """
    from scipy.stats import nbinom
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    if alpha <= 1e-9:
        from scipy.stats import poisson
        return poisson.ppf(tau, mu).astype(float)
    n = 1.0 / alpha
    p = n / (n + mu)
    return nbinom.ppf(tau, n, p).astype(float)
