"""Rolling-origin backtest: the evidence base for every accuracy claim.

Nothing in the repository measured forecast accuracy before this file, so
"improve the forecast" had no yardstick. This supplies one, and applies it
identically to hand-weighted averages, exponential smoothing, intermittent-
demand methods and learned-weight regressions.

PROTOCOL. At each origin week w: train on weeks <= w, forecast week w+1, score
against what actually happened. Nothing at or after w+1 is visible at fit time,
for any model. Four origins are used rather than one so a result cannot be an
artifact of a single lucky week.

WHY THESE ORIGINS. Weeks 34-46. Early weeks are excluded because the deepest lag
window needs 13 weeks of history; the final weeks are excluded because the
relative timeline thins by survivorship and the last few weeks are the thinnest
and least representative. Densification means every series has a row at every
evaluated week, so the evaluation panel is balanced -- the thinning shows up in
exposure, which is exactly where the models should have to deal with it.

METRICS. Reported together because they answer different questions and can
disagree:

  MAE   -- average miss in units. Every SKU counts the same, so 40k tiny SKUs
           dominate it.
  RMSE  -- squared-error, so it penalizes the large misses that cause stockouts.
  WAPE  -- sum|error| / sum|actual|. Volume-weighted, so the SKUs that actually
           drive inventory cost dominate. This is the headline number.
  MASE  -- MAE relative to a seasonal-naive benchmark computed on the TRAINING
           span. Scale-free, so it aggregates across SKUs of wildly different
           volume. < 1 beats naive; > 1 loses to it.
  BIAS  -- mean(forecast - actual). Signed, and the one that matters most here:
           the incumbent's zero-week defect is a bias problem, and bias is what
           propagates into MEIO safety stock. A model can win on MAE and still
           be unusable if it is systematically high.

MAPE is deliberately absent -- it divides by actuals, and 40.5% of the panel
weeks are zero.

SAMPLING. Per-series methods (ETS, ARIMA and friends) are fitted independently
for every series at every origin, which does not scale to 52k x 4 maximum-
likelihood fits. They run on a volume-stratified sample. The global regressions
train on the full panel, but the headline table scores EVERY model on the same
sampled series and the same origins, so the comparison stays apples-to-apples.
"""
import json
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import models as M
from . import panel as P
from . import regressors as R

warnings.filterwarnings("ignore")

ORIGINS = [34, 38, 42, 46]
SAMPLE_SERIES = 4000
SEED = 42
REPORT_PATH = P.ROOT / "reports" / "forecast_benchmark.json"


# ------------------------------------------------------------------ metrics

def mase_scale(train_matrix: np.ndarray) -> np.ndarray:
    """Per-series MASE denominator: mean |y_t - y_{t-1}| over the training span.

    Computed on training data only, so the scale factor itself cannot leak
    information from the evaluation week."""
    d = np.abs(np.diff(train_matrix, axis=1))
    scale = np.nanmean(d, axis=1)
    return np.where(scale > 0, scale, np.nan)


SERVICE_LEVEL = 0.95   # matches meio/build_meio.py's stated network service target


def score(actual: np.ndarray, pred: np.ndarray, scale: np.ndarray,
          tau: float = SERVICE_LEVEL) -> dict:
    """Symmetric accuracy AND asymmetric service metrics, side by side.

    The symmetric block (MAE / RMSE / WAPE / MASE / BIAS) answers "how close is
    this forecast?". It is the wrong question on its own when a stockout costs
    far more than a unit of holding: a perfectly unbiased forecast sits below
    actual demand about half the time, which is the failure mode we are told is
    unacceptable.

    The asymmetric block answers "how often, and by how much, does this forecast
    leave us short?":

      UNDER  -- share of store-weeks where forecast < actual, i.e. the stockout
                frequency this forecast would cause.
      FILL   -- unit fill rate, 1 - shortfall/demand. The FMCG service number:
                of every unit customers wanted, what share could be supplied.
      SHORT  -- total units of unmet demand (the costly side).
      EXCESS -- total units of over-supply (the cheap side; holding cost).
      PINBALL-- quantile loss at tau. The loss function that actually encodes the
                asymmetry: it penalizes shortfall tau/(1-tau) = 19x more heavily
                than excess at tau=0.95, and is minimized by the true tau-th
                quantile of demand rather than its mean.
    """
    err = pred - actual
    abs_err = np.abs(err)
    short = np.clip(actual - pred, 0, None)
    excess = np.clip(pred - actual, 0, None)
    demand = np.nansum(actual)
    with np.errstate(invalid="ignore", divide="ignore"):
        mase_i = abs_err / scale
    return {
        "MAE": float(np.nanmean(abs_err)),
        "RMSE": float(np.sqrt(np.nanmean(err ** 2))),
        "WAPE": float(np.nansum(abs_err) / demand) if demand > 0 else float("nan"),
        "MASE": float(np.nanmean(mase_i)),
        "BIAS": float(np.nanmean(err)),
        "UNDER": float(np.nanmean(pred < actual)),
        "FILL": float(1.0 - np.nansum(short) / demand) if demand > 0 else float("nan"),
        "SHORT": float(np.nansum(short)),
        "EXCESS": float(np.nansum(excess)),
        "PINBALL": float(np.nanmean(np.where(actual >= pred, tau * short, (1 - tau) * excess))),
        "n": int(np.isfinite(actual).sum()),
    }


# ------------------------------------------------------------------ data prep

def stratified_sample(panel: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Sample series stratified by total volume decile, so the per-series arm is
    not accidentally an evaluation of low-volume SKUs only."""
    tot = panel.groupby(["store_id", "product_id"], sort=False)["units"].sum().reset_index()
    tot["decile"] = pd.qcut(tot["units"].rank(method="first"), 10, labels=False)
    per = max(1, n // 10)
    picked = (tot.groupby("decile", group_keys=False)
                 .apply(lambda g: g.sample(min(per, len(g)), random_state=seed)))
    return picked[["store_id", "product_id"]]


def to_matrix(panel: pd.DataFrame):
    """Pivot to a dense [series x week] matrix -- per-series models slice rows of
    this instead of repeatedly grouping a 2.6M-row frame."""
    wide = panel.pivot_table(index=["store_id", "product_id"], columns="rel_week",
                             values="units", aggfunc="sum")
    weeks = np.array(sorted(wide.columns))
    return wide.index, weeks, wide.to_numpy(dtype=float)


# ------------------------------------------------------------------ per-series arm

def _run_series_model(fn, block: np.ndarray) -> np.ndarray:
    """Apply one forecaster to a block of series histories (rows of the panel
    matrix, already truncated at the origin)."""
    out = np.empty(block.shape[0], dtype=float)
    for i in range(block.shape[0]):
        hist = block[i]
        hist = hist[~np.isnan(hist)]
        out[i] = fn(hist) if len(hist) else 0.0
    return out


def per_series_backtest(mat: np.ndarray, weeks: np.ndarray, model_fns: dict,
                        n_jobs: int = -1, n_chunks: int = 24):
    """Every per-series model, every origin, one shared evaluation set.

    Work is dispatched as (model x series-chunk) tasks rather than one task per
    model. ARIMA costs ~48ms per fit against ~0.2ms for a moving average, so
    parallelizing over models alone would leave ARIMA running single-threaded
    while every other worker sat idle.
    """
    results = {name: {"actual": [], "pred": [], "scale": []} for name in model_fns}
    for origin in ORIGINS:
        cut = int(np.searchsorted(weeks, origin, side="right"))     # weeks <= origin
        tgt = int(np.searchsorted(weeks, origin + 1))
        if tgt >= len(weeks) or weeks[tgt] != origin + 1:
            continue
        actual = mat[:, tgt]
        scale = mase_scale(mat[:, :cut])
        hist = mat[:, :cut]
        bounds = np.array_split(np.arange(mat.shape[0]), n_chunks)

        tasks = [(name, fn, rows) for name, fn in model_fns.items() for rows in bounds]
        chunk_preds = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_run_series_model)(fn, hist[rows]) for _, fn, rows in tasks
        )
        acc = {name: [] for name in model_fns}
        for (name, _, _), cp in zip(tasks, chunk_preds):
            acc[name].append(cp)
        for name in model_fns:
            results[name]["actual"].append(actual)
            results[name]["pred"].append(np.concatenate(acc[name]))
            results[name]["scale"].append(scale)
        print(f"    origin {origin}: {len(model_fns)} models x {mat.shape[0]:,} series done")
    return results


# ------------------------------------------------------------------ regression arm

DISPERSION_SAMPLE = 250_000   # rows used to estimate NB overdispersion


def regression_backtest(feat: pd.DataFrame, sample_idx: pd.MultiIndex):
    """Global regressions -- point models, quantile models, and an NB uplift of
    the best point model -- refit at every origin, scored on the sampled series."""
    specs = {**R.make_models(), **R.make_quantile_models()}
    out = {name: {"actual": [], "pred": [], "scale": [], "coef": None} for name in specs}
    for t in R.SERVICE_TAUS:
        out[f"poisson_nb{int(round(t*100))}"] = {"actual": [], "pred": [], "scale": [], "coef": None}

    key = feat.set_index(["store_id", "product_id"]).index
    feat = feat.assign(_in_sample=key.isin(sample_idx))

    for origin in ORIGINS:
        train = feat[feat.rel_week <= origin]
        test = feat[(feat.rel_week == origin + 1) & feat._in_sample]
        if test.empty:
            continue

        tr_eval = train[train._in_sample]
        sc = (tr_eval.sort_values("rel_week")
                     .groupby(["store_id", "product_id"], sort=False)["units"]
                     .apply(lambda s: np.nanmean(np.abs(np.diff(s.to_numpy()))) if len(s) > 1 else np.nan))
        scale = test.set_index(["store_id", "product_id"]).index.map(sc).to_numpy(dtype=float)
        scale = np.where(scale > 0, scale, np.nan)
        actual = test["units"].to_numpy(dtype=float)

        for name, spec in specs.items():
            t0 = time.time()
            pred, coef = R.fit_predict(spec, train, test)
            out[name]["actual"].append(actual)
            out[name]["pred"].append(pred)
            out[name]["scale"].append(scale)
            if coef is not None and out[name]["coef"] is None:
                out[name]["coef"] = coef
            print(f"    origin {origin}: {name:<16} {time.time()-t0:5.1f}s (train {len(train):,} rows)")

            # Turn the calibrated Poisson MEAN into service-level forecasts by
            # reading quantiles off a negative binomial fitted to its own
            # residual spread -- one demand model, every service level.
            if name == "gbm_poisson":
                sub = train.sample(min(DISPERSION_SAMPLE, len(train)), random_state=SEED)
                mu_tr, _ = R.fit_predict({**spec, "est": spec["est"]}, train, sub)
                alpha = R.nb_dispersion(sub["units"].to_numpy(dtype=float), mu_tr)
                print(f"      NB overdispersion alpha = {alpha:.3f} "
                      f"(0 would mean Poisson variance is enough)")
                for t in R.SERVICE_TAUS:
                    nm = f"poisson_nb{int(round(t*100))}"
                    out[nm]["actual"].append(actual)
                    out[nm]["pred"].append(R.nb_quantile(pred, alpha, t))
                    out[nm]["scale"].append(scale)
    return out


def collapse(res: dict) -> dict:
    """Pool every origin into one score per model."""
    table = {}
    for name, d in res.items():
        if not d["actual"]:
            continue
        a = np.concatenate(d["actual"]); p = np.concatenate(d["pred"]); s = np.concatenate(d["scale"])
        m = score(a, p, s)
        if d.get("coef"):
            m["lag_weights"] = {k: v for k, v in d["coef"].items() if k.startswith("lag_")}
            m["all_coefficients"] = d["coef"]
        table[name] = m
    return table


# ------------------------------------------------------------------ runner

def print_table(table: dict, title: str, sort_key: str):
    print(f"\n{title}   (sorted by {sort_key})")
    print(f"{'model':<20} {'MAE':>7} {'WAPE':>7} {'BIAS':>8} | {'UNDER':>7} {'FILL':>7} "
          f"{'SHORT':>9} {'EXCESS':>9} {'PINBALL':>8}")
    print("-" * 96)
    for name, m in sorted(table.items(), key=lambda kv: kv[1][sort_key]):
        flag = "  <-- fails service floor" if m["FILL"] < SERVICE_LEVEL else ""
        print(f"{name:<20} {m['MAE']:>7.3f} {m['WAPE']:>7.4f} {m['BIAS']:>+8.3f} | "
              f"{m['UNDER']:>6.1%} {m['FILL']:>6.1%} {m['SHORT']:>9,.0f} {m['EXCESS']:>9,.0f} "
              f"{m['PINBALL']:>8.4f}{flag}")


def select_model(table: dict, service_level: float = SERVICE_LEVEL):
    """Lowest asymmetric loss among models that clear the service floor.

    An earlier version of this minimized EXCESS units subject to the fill floor.
    That is the wrong objective and the results table shows why: it selected
    gbm_q80, which clears 95% fill on UNITS while being short 15.1% of the time --
    a higher under-rate than the incumbent's 13.4%. Optimizing cost alone walks
    right up to the constraint and buys the cheapest possible compliance,
    trading frequent small shortfalls for a good aggregate number.

    Pinball loss at tau is the loss that actually encodes the asymmetry: it
    charges shortfall tau/(1-tau) times more than excess, so a model cannot buy a
    better score by being short more often. Minimizing it subject to the fill
    floor selects gbm_q90 instead -- 97.6% fill at a 7.4% under-rate, roughly
    half the incumbent's shortfall frequency.

    Excess units remain in the table and are reported for the selected model,
    because the cost of that protection is a real number the business should see
    rather than one the selection rule quietly minimized on its behalf.
    """
    feasible = {k: v for k, v in table.items() if v["FILL"] >= service_level}
    if not feasible:
        return None, min(table.items(), key=lambda kv: kv[1]["PINBALL"])[0]
    return min(feasible.items(), key=lambda kv: kv[1]["PINBALL"])[0], None


def main():
    t_start = time.time()
    con = duckdb.connect(str(P.DB_PATH), read_only=True)
    print("Loading dense panel...")
    pan = P.load_panel(con, min_weeks=14)
    con.close()
    print(f"  {len(pan):,} rows / {pan.groupby(['store_id','product_id']).ngroups:,} series")

    print(f"\nSampling {SAMPLE_SERIES:,} series (volume-stratified) for the per-series arm...")
    samp = stratified_sample(pan, SAMPLE_SERIES)
    sample_idx = pd.MultiIndex.from_frame(samp)
    sub = pan.merge(samp, on=["store_id", "product_id"], how="inner")
    idx, weeks, mat = to_matrix(sub)
    print(f"  matrix {mat.shape[0]:,} series x {mat.shape[1]} weeks")

    print(f"\nPer-series models at origins {ORIGINS}...")
    t0 = time.time()
    ps_res = per_series_backtest(mat, weeks, {**M.PER_SERIES_MODELS, **M.SLOW_MODELS})
    print(f"  done in {time.time()-t0:.1f}s")

    print("\nBuilding regression features on the full panel...")
    t0 = time.time()
    feat = R.build_features(pan)
    print(f"  {len(feat):,} rows x {len(R.FEATURES)} features in {time.time()-t0:.1f}s")

    print("\nGlobal regression + quantile models...")
    rg_res = regression_backtest(feat, sample_idx)

    table = {**collapse(ps_res), **collapse(rg_res)}

    print_table(table, f"ACCURACY VIEW -- {mat.shape[0]:,} series x {len(ORIGINS)} origins", "WAPE")
    print_table(table, f"SERVICE VIEW  -- asymmetric cost, floor = {SERVICE_LEVEL:.0%} fill", "PINBALL")

    inc = table["wma_rows_incumbent"]
    chosen, fallback = select_model(table)
    print("\n" + "=" * 96)
    if chosen is None:
        print(f"NO model reaches {SERVICE_LEVEL:.0%} fill. Best on pinball loss: {fallback}")
        chosen = fallback
    best = table[chosen]
    print(f"SELECTED: {chosen}")
    print(f"  fill rate    {inc['FILL']:.2%}  ->  {best['FILL']:.2%}")
    print(f"  under-rate   {inc['UNDER']:.2%}  ->  {best['UNDER']:.2%}")
    print(f"  short units  {inc['SHORT']:,.0f}  ->  {best['SHORT']:,.0f}   "
          f"({(best['SHORT']/inc['SHORT']-1)*100:+.1f}% change)")
    print(f"  excess units {inc['EXCESS']:,.0f}  ->  {best['EXCESS']:,.0f}   "
          f"({(best['EXCESS']/inc['EXCESS']-1)*100:+.1f}% change  <- the cost of the protection)")
    print(f"  WAPE         {inc['WAPE']:.4f}  ->  {best['WAPE']:.4f}")
    print("=" * 96)

    for name in ("ridge", "lasso"):
        if name in table and "lag_weights" in table[name]:
            lw = table[name]["lag_weights"]
            tot = sum(abs(v) for v in lw.values()) or 1.0
            print(f"\nLearned lag weights -- {name} (raw scale, and share of total |b|):")
            for k, v in lw.items():
                print(f"  {k:<8} {v:>9.4f}   {abs(v)/tot:>6.1%}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "protocol": {
            "origins": ORIGINS, "horizon_weeks": 1,
            "sampled_series": int(mat.shape[0]), "sampling": "volume-stratified deciles",
            "eval_points_per_model": int(table["naive"]["n"]),
            "panel_rows_total": int(len(pan)),
            "service_level_floor": SERVICE_LEVEL,
            "selection_rule": "min PINBALL loss subject to FILL >= service floor",
            "selected_model": chosen,
            "metrics_note": ("MAPE omitted: 40.5% of panel weeks are zero. "
                             "PINBALL is the asymmetric loss actually optimized."),
        },
        "results": table,
    }, indent=2))
    print(f"\nWrote {REPORT_PATH.relative_to(P.ROOT)}   [{time.time()-t_start:.1f}s]")


if __name__ == "__main__":
    main()
