"""Predict how accurate the forecast will be, per SKU, BEFORE observing the error.

The benchmark reports one accuracy number for the whole network. That average
hides the fact that a smooth weekly seller and a lumpy SKU that moves 40 units
once a quarter are not remotely equally predictable. If we can say in advance
which store-SKUs the forecast will handle badly, protection can be aimed at them
instead of spread evenly -- which matters directly here, because under-supply is
the expensive error and uniform safety stock buys it in the wrong places.

THE CONSTRUCTION. This is a second regression stacked on the first:

    stage 1   demand model      features(t-1..t-13)   ->  demand at t
    stage 2   accuracy model    series descriptors    ->  |error| of stage 1

Stage 2's TARGET is stage 1's realized error, measured on held-out weeks by the
same rolling-origin protocol. Its INPUTS are descriptors of the series computed
strictly from the training window, so a prediction is available before the
forecast week happens -- which is the entire point. A model that needed the
error to predict the error would be useless.

DESCRIPTORS. The classical forecastability pair, from the Syntetos-Boylan-Croston
literature:

  ADI  average demand interval -- weeks between non-zero demands. High ADI means
       intermittent: the timing is the hard part.
  CV2  squared coefficient of variation of NON-ZERO demand sizes. High CV2 means
       erratic: the size is the hard part.

Split at the conventional ADI=1.32 / CV2=0.49 cutoffs these give four regimes --
smooth, erratic, intermittent, lumpy -- and the ordering of realized error across
those four is a sanity check on the whole exercise: if lumpy series do not
forecast worse than smooth ones, something is wrong upstream. Volume, history
length, recent trend and demand entropy are supplied alongside, because the
classical pair is a 2-D summary of something richer.

VALIDATION. Predicting error is only useful if the prediction holds out of
sample, so the meta-model trains on the early origins and is tested on the last
one. Reported: Spearman rank correlation (can it RANK which SKUs will go wrong --
the operationally relevant question), R^2 on log error, and decile lift (how much
worse is the worst predicted decile than the best). Rank correlation leads
because the policy use is ordering SKUs by risk, not hitting an error value.

OUTPUT. A per-SKU predicted error and risk decile, written to
`forecast_accuracy_prediction`, plus a recommended per-SKU service level: high-
risk SKUs are pushed to a stricter tau and low-risk SKUs relaxed, so service is
bought where shortfall is actually likely rather than uniformly.
"""
import json
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import panel as P
from . import regressors as R

ORIGINS = [34, 38, 42, 46]
TEST_ORIGIN = 46           # held out from the meta-model entirely
REPORT_PATH = P.ROOT / "reports" / "forecastability.json"

ADI_CUT, CV2_CUT = 1.32, 0.49
DESCRIPTORS = ["adi", "cv2", "mean_units", "total_units", "zero_share",
               "n_weeks", "recent_trend", "demand_entropy", "max_gap", "std_units"]


def sbc_class(adi: float, cv2: float) -> str:
    """Syntetos-Boylan-Croston regime. Named rather than numeric because the
    label is what a planner acts on."""
    if adi < ADI_CUT:
        return "smooth" if cv2 < CV2_CUT else "erratic"
    return "intermittent" if cv2 < CV2_CUT else "lumpy"


def series_descriptors(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-series forecastability descriptors, computed on the TRAINING window
    only so they are knowable before the forecast week."""
    def one(g: pd.DataFrame) -> pd.Series:
        u = g["units"].to_numpy(dtype=float)
        nz = u[u > 0]
        nz_idx = np.flatnonzero(u > 0)
        adi = float(np.mean(np.diff(nz_idx))) if len(nz_idx) > 1 else float(len(u))
        cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 and nz.mean() > 0 else 0.0
        half = max(len(u) // 2, 1)
        recent, early = u[-half:].mean(), u[:half].mean()
        # Shannon entropy of the demand-share distribution: a flat series across
        # weeks scores high, one dominated by a single spike scores low.
        tot = u.sum()
        if tot > 0:
            pk = u / tot
            pk = pk[pk > 0]
            ent = float(-np.sum(pk * np.log(pk)) / np.log(len(u))) if len(u) > 1 else 0.0
        else:
            ent = 0.0
        gaps = np.diff(nz_idx) if len(nz_idx) > 1 else np.array([len(u)])
        return pd.Series({
            "adi": adi, "cv2": cv2,
            "mean_units": float(u.mean()), "std_units": float(u.std()),
            "total_units": float(tot), "zero_share": float((u == 0).mean()),
            "n_weeks": float(len(u)),
            "recent_trend": float(recent / early) if early > 0 else 1.0,
            "demand_entropy": ent, "max_gap": float(gaps.max()),
        })

    d = hist.groupby(["store_id", "product_id"], sort=False).apply(
        one, include_groups=False).reset_index()
    d["sbc_class"] = [sbc_class(a, c) for a, c in zip(d["adi"], d["cv2"])]
    return d


def collect_errors(pan: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Stage-1 errors on held-out weeks, per series per origin, over the FULL
    panel -- the labels the accuracy model learns from."""
    spec = R.make_models()["gbm_poisson"]
    rows = []
    for origin in ORIGINS:
        train = feat[feat.rel_week <= origin]
        test = feat[feat.rel_week == origin + 1]
        if test.empty:
            continue
        t0 = time.time()
        pred, _ = R.fit_predict(spec, train, test)
        actual = test["units"].to_numpy(dtype=float)
        rows.append(pd.DataFrame({
            "store_id": test["store_id"].to_numpy(),
            "product_id": test["product_id"].to_numpy(),
            "origin": origin,
            "actual": actual, "pred": pred,
            "abs_err": np.abs(pred - actual),
            "shortfall": np.clip(actual - pred, 0, None),
        }))
        print(f"  origin {origin}: stage-1 fitted + scored in {time.time()-t0:.1f}s "
              f"({len(test):,} series)")
    return pd.concat(rows, ignore_index=True)


def main():
    t_start = time.time()
    con = duckdb.connect(str(P.DB_PATH))
    print("Loading dense panel...")
    pan = P.load_panel(con, min_weeks=14)
    print(f"  {len(pan):,} rows / {pan.groupby(['store_id','product_id']).ngroups:,} series")

    print("Building stage-1 features...")
    feat = R.build_features(pan)

    print("\nCollecting stage-1 errors (the accuracy model's target)...")
    err = collect_errors(pan, feat)

    print("\nComputing series descriptors per origin (training window only)...")
    parts = []
    for origin in ORIGINS:
        d = series_descriptors(pan[pan.rel_week <= origin])
        d["origin"] = origin
        parts.append(d)
        print(f"  origin {origin}: {len(d):,} series described")
    desc = pd.concat(parts, ignore_index=True)

    data = err.merge(desc, on=["store_id", "product_id", "origin"], how="inner")
    data["y"] = np.log1p(data["abs_err"])      # log target: errors are heavily skewed

    train = data[data.origin != TEST_ORIGIN]
    test = data[data.origin == TEST_ORIGIN]
    print(f"\nMeta-model: train {len(train):,} rows (origins "
          f"{[o for o in ORIGINS if o != TEST_ORIGIN]}) -> test {len(test):,} rows (origin {TEST_ORIGIN})")

    Xtr, ytr = train[DESCRIPTORS].to_numpy(dtype=float), train["y"].to_numpy()
    Xte, yte = test[DESCRIPTORS].to_numpy(dtype=float), test["y"].to_numpy()
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)

    meta = {}
    for name, est in {
        "ridge": Ridge(alpha=1.0),
        "gbm": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                             max_leaf_nodes=31, random_state=42),
    }.items():
        if name == "ridge":
            mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
            est.fit((Xtr - mu) / sd, ytr)
            pr = est.predict((Xte - mu) / sd)
        else:
            est.fit(Xtr, ytr)
            pr = est.predict(Xte)
        rho = float(spearmanr(pr, yte).statistic)
        ss_res = float(np.sum((yte - pr) ** 2))
        ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        meta[name] = {"spearman": rho, "r2_log": 1 - ss_res / ss_tot,
                      "pred": pr}
        print(f"  {name:<6} Spearman rho = {rho:+.3f}   R2(log err) = {1-ss_res/ss_tot:+.3f}")

    best_name = max(meta, key=lambda k: meta[k]["spearman"])
    pred = meta[best_name]["pred"]
    print(f"\nBest accuracy model: {best_name}")

    # Decile lift: does the top predicted-risk decile actually forecast worse?
    t = test.copy()
    t["pred_err"] = np.expm1(pred)
    t["decile"] = pd.qcut(t["pred_err"].rank(method="first"), 10, labels=False)
    lift = t.groupby("decile").agg(
        predicted_err=("pred_err", "mean"), actual_err=("abs_err", "mean"),
        actual_shortfall=("shortfall", "mean"), n=("abs_err", "size"))
    print("\nDecile lift on held-out origin (0 = predicted easiest, 9 = predicted hardest):")
    print(lift.round(3).to_string())
    d0, d9 = float(lift["actual_err"].iloc[0]), float(lift["actual_err"].iloc[-1])
    print(f"\n  Actual error, hardest decile vs easiest: {d9:.3f} vs {d0:.3f}  ({d9/max(d0,1e-9):.1f}x)")

    print("\nRealized error by Syntetos-Boylan-Croston regime (held-out origin):")
    by_class = t.groupby("sbc_class").agg(
        n=("abs_err", "size"), mean_demand=("actual", "mean"),
        actual_err=("abs_err", "mean"), shortfall=("shortfall", "mean"),
        adi=("adi", "mean"), cv2=("cv2", "mean"))
    by_class["wape"] = t.groupby("sbc_class").apply(
        lambda g: g["abs_err"].sum() / max(g["actual"].sum(), 1e-9), include_groups=False)
    print(by_class.round(3).to_string())

    # Risk-weighted service levels: spend protection where shortfall is likely.
    base, spread = 0.95, 0.04
    t["recommended_tau"] = np.clip(base + spread * (t["decile"] - 4.5) / 4.5, 0.90, 0.995)
    rec = (t.groupby(["store_id", "product_id"])
             .agg(predicted_abs_error=("pred_err", "mean"),
                  risk_decile=("decile", "max"),
                  sbc_class=("sbc_class", "last"),
                  adi=("adi", "mean"), cv2=("cv2", "mean"),
                  recommended_tau=("recommended_tau", "mean"))
             .reset_index())
    con.register("acc_pred", rec)
    con.execute("CREATE OR REPLACE TABLE forecast_accuracy_prediction AS SELECT * FROM acc_pred")
    print(f"\nWrote table forecast_accuracy_prediction ({len(rec):,} series)")

    REPORT_PATH.write_text(json.dumps({
        "design": {
            "stage1": "gbm_poisson demand model",
            "stage2_target": "log1p(|stage-1 error|) on held-out weeks",
            "descriptors": DESCRIPTORS,
            "train_origins": [o for o in ORIGINS if o != TEST_ORIGIN],
            "test_origin": TEST_ORIGIN,
            "sbc_cutoffs": {"ADI": ADI_CUT, "CV2": CV2_CUT},
        },
        "meta_model": {k: {"spearman": v["spearman"], "r2_log": v["r2_log"]}
                       for k, v in meta.items()},
        "best_meta_model": best_name,
        "decile_lift": lift.reset_index().to_dict(orient="records"),
        "by_sbc_class": by_class.reset_index().to_dict(orient="records"),
        "hardest_vs_easiest_decile_error_ratio": d9 / max(d0, 1e-9),
    }, indent=2, default=float))
    con.close()
    print(f"Wrote {REPORT_PATH.relative_to(P.ROOT)}   [{time.time()-t_start:.1f}s]")


if __name__ == "__main__":
    main()
