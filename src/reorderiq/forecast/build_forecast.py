"""Fit the selected model on all history and write the production forecast.

Reads the winner chosen by backtest.py (reports/forecast_benchmark.json) rather
than hard-coding a model name, so the thing that ships is the thing the
rolling-origin evaluation actually selected.

SCHEMA. `forecast_store_product` keeps every column it had, so meio/build_meio.py
and agents/control_tower.py continue to run untouched. Added alongside:

  forecast_units_mean   calibrated conditional mean -- the unbiased estimate, for
                        anything that needs an expectation rather than a service
                        target (financial planning, capacity, aggregation).
  forecast_units_p{tau} the service-level frontier, one column per tau, so the
                        service level can be re-chosen without refitting.
  hist_std_dense        standard deviation over the DENSIFIED series.

`hist_std` is preserved as it was, and `hist_std_dense` added next to it rather
than replacing it, because MEIO multiplies that column by z=1.645 to size safety
stock. The original is computed over non-zero weeks only, so it measures
variability *given a sale happened* -- a different quantity from the weekly
demand sigma the safety-stock formula assumes. Swapping it silently would move
every reorder point in the network inside a commit about forecasting; exposing
both lets that be its own decision, with the ratio printed at the end.

BIAS IS NOW DELIBERATE. `forecast_units_next_week` carries the service-level
forecast, not the mean, because shortfall is the expensive error here. That is a
policy choice, and it is recorded in `forecast_metadata` alongside the tau it was
produced at -- so the high side is documented and tunable rather than being an
accident of a windowing bug, which is what it was before.
"""
import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from . import panel as P
from . import regressors as R

BENCHMARK_PATH = P.ROOT / "reports" / "forecast_benchmark.json"
SUMMARY_PATH = P.ROOT / "reports" / "forecast_summary.json"
MODEL_PATH = P.ROOT / "data" / "processed" / "forecast_model.joblib"
TARGET_WEEK = P.LAST_WEEK + 1


def selected_model_name(default: str = "gbm_q95") -> str:
    if BENCHMARK_PATH.exists():
        meta = json.loads(BENCHMARK_PATH.read_text()).get("protocol", {})
        return meta.get("selected_model") or default
    return default


def main():
    t0 = time.time()
    con = duckdb.connect(str(P.DB_PATH))

    chosen = selected_model_name()
    print(f"Selected model from backtest: {chosen}")

    print("Loading dense panel...")
    pan = P.load_panel(con, min_weeks=2)
    print(f"  {len(pan):,} rows / {pan.groupby(['store_id','product_id']).ngroups:,} series")

    # Train on everything; predict the week after the last observed one. The
    # feature row for TARGET_WEEK is built by appending one empty week per
    # series, so its lags are the real observed weeks 52, 51, ... -- nothing
    # about the target week itself is needed. Features are built once over the
    # extended frame and then split: a week <= 52 row has identical features
    # either way, since every window looks strictly backwards.
    last = pan.groupby(["store_id", "product_id"], as_index=False).agg(
        department_id=("department_id", "last"), active_users=("active_users", "last"))
    last["rel_week"] = TARGET_WEEK
    last["units"] = np.nan
    print("Building features...")
    feat = R.build_features(pd.concat([pan, last[pan.columns]], ignore_index=True))
    test = feat[feat.rel_week == TARGET_WEEK].copy()
    train = feat[feat.rel_week <= P.LAST_WEEK]
    print(f"  train {len(train):,} rows -> predict {len(test):,} series for week {TARGET_WEEK}")

    specs = {**R.make_models(), **R.make_quantile_models()}
    out = test[["store_id", "department_id", "product_id"]].copy()

    # Mean forecast: needed on its own merits, and it is the base the NB uplift
    # reads service levels off.
    print("Fitting mean model (gbm_poisson)...")
    mean_spec = specs["gbm_poisson"]
    mean_pred, _ = R.fit_predict(mean_spec, train, test)
    out["forecast_units_mean"] = mean_pred

    # Dispersion of the mean model's own residuals, needed to turn a mean into a
    # service level at any horizon (see agents/demand_projection.py).
    disp_sub = train.sample(min(250_000, len(train)), random_state=42)
    mu_tr, _ = R.fit_predict(mean_spec, train, disp_sub)
    nb_alpha = R.nb_dispersion(disp_sub["units"].to_numpy(dtype=float), mu_tr)
    print(f"  NB overdispersion alpha = {nb_alpha:.3f}")

    print("Fitting quantile models...")
    for tau in R.SERVICE_TAUS:
        nm = f"gbm_q{int(round(tau*100))}"
        pred, _ = R.fit_predict(specs[nm], train, test)
        out[f"forecast_units_p{int(round(tau*100))}"] = pred
        print(f"  {nm}: mean {pred.mean():.2f} units")

    # The shipped column.
    if chosen in specs and chosen.startswith("gbm_q"):
        ship = out[f"forecast_units_p{chosen.replace('gbm_q','')}"].to_numpy()
    elif chosen.startswith("poisson_nb"):
        tau = int(chosen.replace("poisson_nb", "")) / 100.0
        ship = R.nb_quantile(mean_pred, nb_alpha, tau)
    else:
        ship, _ = R.fit_predict(specs[chosen], train, test)
    out["forecast_units_next_week"] = np.clip(ship, 0.0, None)

    # History stats. hist_mean/hist_std/n_weeks_observed keep their original
    # sparse definitions so downstream consumers are byte-compatible; the dense
    # equivalents are added alongside.
    dense = (pan.groupby(["store_id", "product_id"], as_index=False)
                .agg(hist_mean_dense=("units", "mean"),
                     hist_std_dense=("units", "std"),
                     n_weeks_panel=("units", "size")))
    sparse = con.execute("""
        SELECT store_id, product_id, avg(units) AS hist_mean, stddev_samp(units) AS hist_std,
               count(*) AS n_weeks_observed
        FROM demand_store_product_week GROUP BY 1, 2
    """).fetchdf()
    out = out.merge(dense, on=["store_id", "product_id"], how="left") \
             .merge(sparse, on=["store_id", "product_id"], how="left")
    out["hist_std"] = out["hist_std"].fillna(0.0)
    out["hist_std_dense"] = out["hist_std_dense"].fillna(0.0)
    out["n_weeks_observed"] = out["n_weeks_observed"].fillna(0).astype("int64")

    print("\nWriting tables...")
    con.execute("CREATE OR REPLACE TABLE forecast_baseline AS SELECT * FROM forecast_store_product")
    n_base = con.execute("SELECT count(*) FROM forecast_baseline").fetchone()[0]
    print(f"  forecast_baseline: preserved {n_base:,} rows of the original forecast")

    con.register("new_forecast", out)
    con.execute("""
        CREATE OR REPLACE TABLE forecast_store_product AS
        SELECT store_id, department_id, product_id,
               forecast_units_next_week, hist_mean, hist_std, n_weeks_observed,
               forecast_units_mean, forecast_units_p80, forecast_units_p90,
               forecast_units_p95, forecast_units_p99,
               hist_mean_dense, hist_std_dense, n_weeks_panel
        FROM new_forecast
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_fsp_store ON forecast_store_product(store_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fsp_product ON forecast_store_product(product_id)")

    con.execute("DROP TABLE IF EXISTS forecast_metadata")
    con.execute("""
        CREATE TABLE forecast_metadata (
            shipped_model VARCHAR, shipped_quantity VARCHAR, service_level DOUBLE,
            target_rel_week INTEGER, built_at TIMESTAMP
        )
    """)
    tau_ship = float(chosen.replace("gbm_q", "").replace("poisson_nb", "")) / 100.0 \
        if chosen.startswith(("gbm_q", "poisson_nb")) else float("nan")
    con.execute("INSERT INTO forecast_metadata VALUES (?, ?, ?, ?, now())",
                [chosen, "service-level quantile" if tau_ship == tau_ship else "conditional mean",
                 tau_ship, TARGET_WEEK])

    # ---- comparison against what was there before ----
    cmp = con.execute("""
        SELECT sum(b.forecast_units_next_week) AS baseline_units,
               sum(f.forecast_units_next_week) AS new_units,
               sum(f.forecast_units_mean)      AS new_mean_units,
               avg(b.hist_std)                 AS baseline_sigma,
               avg(f.hist_std_dense)           AS dense_sigma
        FROM forecast_store_product f
        JOIN forecast_baseline b USING (store_id, product_id)
    """).fetchdf()
    b_u, n_u, m_u = (float(cmp[c][0]) for c in ("baseline_units", "new_units", "new_mean_units"))
    b_s, d_s = float(cmp["baseline_sigma"][0]), float(cmp["dense_sigma"][0])

    print(f"\n  Network next-week units")
    print(f"    baseline (as shipped)      {b_u:>14,.0f}")
    print(f"    new mean forecast          {m_u:>14,.0f}   ({m_u/b_u-1:+.1%} vs baseline)")
    print(f"    new shipped ({chosen})      {n_u:>14,.0f}   ({n_u/b_u-1:+.1%} vs baseline)")
    print(f"\n  Sigma feeding MEIO safety stock")
    print(f"    hist_std (sparse, in use)  {b_s:>14,.2f}")
    print(f"    hist_std_dense (available) {d_s:>14,.2f}   ({d_s/b_s-1:+.1%})")

    SUMMARY_PATH.write_text(json.dumps({
        "shipped_model": chosen, "service_level": tau_ship, "target_rel_week": TARGET_WEEK,
        "n_series": int(len(out)),
        "network_units": {"baseline": b_u, "new_mean": m_u, "new_shipped": n_u},
        "sigma": {"hist_std_sparse": b_s, "hist_std_dense": d_s},
        "note": ("forecast_units_next_week is a service-level quantile, not a mean. "
                 "MEIO adds safety stock on top of it, so protection compounds; "
                 "switch MEIO to forecast_units_mean to avoid double-counting."),
    }, indent=2))
    # Persist the fitted mean model so multi-week projection can recurse with it
    # instead of re-deriving a second, inconsistent forecasting method.
    joblib.dump({"model": mean_spec["est"], "features": mean_spec["features"],
                 "kind": mean_spec["kind"], "nb_alpha": nb_alpha,
                 "trained_through_week": P.LAST_WEEK, "shipped_model": chosen,
                 "service_level": tau_ship}, MODEL_PATH)
    print(f"Wrote {MODEL_PATH.relative_to(P.ROOT)}")

    con.close()
    print(f"\nWrote {SUMMARY_PATH.relative_to(P.ROOT)}   [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
