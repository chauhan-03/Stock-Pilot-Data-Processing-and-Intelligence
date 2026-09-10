"""Demand Projection Agent: multi-week demand projection for a store /
department / product scope, at a chosen service level.

Supersedes the earlier version of this file, which recursively re-applied the
pipeline's 6-week weighted moving average. Three things were wrong with that,
all measured in reports/forecast_benchmark.json:

 1. The moving average it extended took the last 6 ROWS of a table that only
    records weeks demand occurred, so in 80.4% of series those rows spanned
    more than 6 weeks and zero-demand weeks never entered the average. Measured
    against held-out weeks it over-forecast by +68%.
 2. It projected a point estimate with no notion of service level, so a planner
    could not tell whether a number was a central estimate or a safe one.
 3. Feeding a forecast back into itself flattens the series toward a constant.

WHAT THIS DOES INSTEAD. It recurses the fitted MEAN model saved by
forecast/build_forecast.py, then converts to a service level only at read-out.
That ordering is deliberate and is the substantive fix:

    recurse the mean  ->  apply the quantile at each horizon

Recursing a high-sided forecast instead would compound the uplift -- week 4
would be the 95th percentile of a series already inflated by three 95th-
percentile steps, which is not a 95% service level but an unbounded one. The
mean path is the only self-consistent thing to feed back; the asymmetry belongs
at the point of use.

UNCERTAINTY GROWS WITH HORIZON. An h-week-ahead forecast is strictly less
certain than a 1-week-ahead one, because from week 2 onward part of the lag
window is itself predicted rather than observed. The service-level band is
widened accordingly, scaling the negative-binomial dispersion linearly in h.
Linear growth is the random-walk approximation, not an exact result for this
model -- it is a documented rule of thumb, and the honest reading is that far
horizons are wider than the near ones by roughly the right order, not that the
week-8 interval is calibrated to the decimal.

Still a plain function, not an LLM agent, for the same reason as
agents/control_tower.py: the scope filters are unambiguous and the projection is
deterministic. An LLM layer could turn "will store 12 run short on dairy next
month?" into these arguments and narrate the answer, but it would add nothing to
the computation.
"""
import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "processed" / "reorderiq.duckdb"
MODEL_PATH = ROOT / "data" / "processed" / "forecast_model.joblib"

import sys
sys.path.insert(0, str(ROOT / "src"))
from reorderiq.forecast import panel as P          # noqa: E402
from reorderiq.forecast import regressors as R     # noqa: E402

DEFAULT_SERVICE_LEVEL = 0.95


def _load_scope(con, store_id, department_id, product_id) -> pd.DataFrame:
    filters, params = [], []
    for col, val in [("store_id", store_id), ("department_id", department_id),
                     ("product_id", product_id)]:
        if val is not None:
            filters.append(f"{col} = ?")
            params.append(val)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    return con.execute(f"""
        SELECT store_id, department_id, product_id, rel_week, units, active_users
        FROM demand_panel {where}
        ORDER BY store_id, department_id, product_id, rel_week
    """, params).fetchdf()


def project_demand(con, store_id=None, department_id=None, product_id=None,
                   horizon_weeks: int = 4, service_level: float = DEFAULT_SERVICE_LEVEL) -> dict:
    """Project demand `horizon_weeks` ahead for the given scope.

    Any filter left as None widens the scope -- a department with no store_id
    projects that department across every store carrying it.

    Returns both the mean path and the service-level path. The mean is what to
    use for planning volumes and financials; the service-level path is what to
    use for cover, since under-supply costs more than over-supply.
    """
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be >= 1")
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level must be in [0.5, 1.0)")

    scope = {"store_id": store_id, "department_id": department_id, "product_id": product_id}
    df = _load_scope(con, store_id, department_id, product_id)
    if df.empty:
        return {"scope": scope, "horizon_weeks": horizon_weeks, "n_series_projected": 0,
                "network_total_by_week": [], "series": [],
                "note": "No matching store-product demand history for this scope."}

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found -- run forecast/build_forecast.py first.")
    bundle = joblib.load(MODEL_PATH)
    spec = {"est": bundle["model"], "features": bundle["features"],
            "kind": bundle["kind"], "scale": False}
    alpha1 = float(bundle["nb_alpha"])

    names = con.execute("SELECT product_id, product_name FROM products").fetchdf()
    name_lookup = dict(zip(names["product_id"], names["product_name"]))
    depts = con.execute("SELECT department_id, department FROM departments").fetchdf()
    dept_lookup = dict(zip(depts["department_id"], depts["department"]))

    keys = ["store_id", "department_id", "product_id"]
    work = df.copy()
    mean_paths, svc_paths = [], []

    # Recurse the mean one week at a time: predict, append the prediction as the
    # new "observed" week, rebuild features, repeat.
    for h in range(1, horizon_weeks + 1):
        nxt = (work.sort_values("rel_week").groupby(keys, as_index=False)
                   .agg(active_users=("active_users", "last"), rel_week=("rel_week", "max")))
        nxt["rel_week"] += 1
        nxt["units"] = np.nan
        stacked = pd.concat([work, nxt[work.columns]], ignore_index=True)

        feats = R.build_features(stacked)
        target_week = int(nxt["rel_week"].iloc[0])
        test = feats[feats.rel_week == target_week].copy()
        mu = R.predict_only(spec, test)

        # Uncertainty compounds with horizon: from h=2 the lag window contains
        # predictions, not observations.
        svc = R.nb_quantile(mu, alpha1 * h, service_level)

        step = test[keys].copy()
        step["h"] = h
        step["mean"] = mu
        step["service"] = svc
        mean_paths.append(step)

        # Feed back the MEAN, never the service-level value -- see module docstring.
        fed = test[keys + ["rel_week", "active_users"]].copy()
        fed["units"] = mu
        work = pd.concat([work, fed[work.columns]], ignore_index=True)

    allsteps = pd.concat(mean_paths, ignore_index=True)
    wide_mean = allsteps.pivot_table(index=keys, columns="h", values="mean")
    wide_svc = allsteps.pivot_table(index=keys, columns="h", values="service")

    series = []
    last_obs = df.sort_values("rel_week").groupby(keys)["units"].last()
    for k in wide_mean.index:
        sid, did, pid = k
        series.append({
            "store_id": int(sid), "department_id": int(did),
            "department": dept_lookup.get(did, "?"),
            "product_id": int(pid), "product_name": name_lookup.get(pid, "?"),
            "last_observed_units": float(last_obs.get(k, np.nan)),
            "projected_mean": [round(float(v), 1) for v in wide_mean.loc[k].to_numpy()],
            "projected_service": [round(float(v), 1) for v in wide_svc.loc[k].to_numpy()],
        })
    series.sort(key=lambda s: -sum(s["projected_service"]))

    return {
        "scope": scope,
        "horizon_weeks": horizon_weeks,
        "service_level": service_level,
        "n_series_projected": len(series),
        "network_mean_by_week": [round(float(v), 1) for v in wide_mean.sum(axis=0).to_numpy()],
        "network_service_by_week": [round(float(v), 1) for v in wide_svc.sum(axis=0).to_numpy()],
        "series": series,
        "method": (f"recursive mean projection with {bundle['shipped_model']}'s companion "
                   f"mean model, converted to a P{service_level:.0%} service level per horizon "
                   f"via negative-binomial quantiles (dispersion scaled linearly in h)"),
        "caveat": ("Horizon uncertainty compounds: from week 2 the lag window contains "
                   "predictions, not observations. Bands widen with h by a random-walk "
                   "approximation, so far horizons are indicative rather than calibrated."),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Project demand for a store/department/product scope at a service level.")
    ap.add_argument("--store", type=int, default=None, dest="store_id")
    ap.add_argument("--department", type=int, default=None, dest="department_id")
    ap.add_argument("--product", type=int, default=None, dest="product_id")
    ap.add_argument("--weeks", type=int, default=4, dest="horizon_weeks")
    ap.add_argument("--service-level", type=float, default=DEFAULT_SERVICE_LEVEL)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    res = project_demand(con, args.store_id, args.department_id, args.product_id,
                         args.horizon_weeks, args.service_level)
    con.close()

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print(f"Scope: {res['scope']}  |  {res['n_series_projected']:,} series  |  "
          f"{res['horizon_weeks']} weeks ahead")
    if res["n_series_projected"] == 0:
        print(res["note"])
        return
    print(f"Method: {res['method']}\n")
    print(f"Network mean by week    : {res['network_mean_by_week']}")
    print(f"Network P{args.service_level:.0%} by week    : {res['network_service_by_week']}")
    print(f"\nTop {min(args.top, len(res['series']))} series by projected cover requirement:")
    for s in res["series"][:args.top]:
        print(f"  store {s['store_id']:>3} | {s['department']:<14} | {s['product_name'][:30]:<30} "
              f"| last {s['last_observed_units']:>5.0f} | mean {s['projected_mean']} "
              f"| P{args.service_level:.0%} {s['projected_service']}")
    print(f"\n{res['caveat']}")


if __name__ == "__main__":
    main()
