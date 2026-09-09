"""Model 1: Logistic regression on reorder probability.

Target: reordered (binary)
Features: The same 16 engineered features used by HistGradientBoostingClassifier
Split: User-level 90/10, matching HGB validation split for fair comparison

Outputs:
  - Odds ratios with 95% CIs, sorted by effect size
  - Coefficient p-values and standard errors
  - McFadden pseudo-R²
  - ROC-AUC on held-out set, delta vs HGB's 0.831
  - Confusion matrix at optimal threshold
  - Plain-English interpretation table
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .data import load_dataset, train_val_split, FEATURES, CONTINUOUS_FEATURES
from .diagnostics import VIFDiagnostic, SeparationTest, BoxTidwellTest

ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = ROOT / "reports" / "regression"

logger = logging.getLogger(__name__)

HGB_AUC = 0.831  # Baseline from train_reorder_model.py


class LogisticRegressionModel:
    """Logistic regression on reorder probability."""

    def __init__(self, sample_size=None, random_seed=42):
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.model = None
        self.result = None
        self.threshold = None
        self.diagnostics = {}
        self.metrics = {}

    def load_and_split(self):
        """Load data and apply user-level 90/10 split."""
        logger.info("Loading dataset...")
        df = load_dataset(sample_size=self.sample_size, random_seed=self.random_seed)
        
        logger.info("Applying user-level 90/10 split...")
        X_train, X_val, y_train, y_val, val_users, (n_train, n_val) = train_val_split(
            df, val_fraction=0.1, random_seed=self.random_seed
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.val_users = val_users
        
        return {
            "train_rows": n_train,
            "val_rows": n_val,
            "val_users": len(val_users),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
        }

    def run_diagnostics(self):
        """Run all pre-fit diagnostics."""
        logger.info("Running diagnostics...")
        
        diags = {}
        
        # 1. VIF
        logger.info("  Computing VIF...")
        X_numeric = self.X_train[CONTINUOUS_FEATURES]
        vif_df = VIFDiagnostic.compute(X_numeric, CONTINUOUS_FEATURES)
        vif_report = VIFDiagnostic.report(vif_df)
        diags["vif"] = vif_report
        logger.info(f"    Max VIF: {vif_report['max_vif']:.2f} (violations: {vif_report['n_violations']})")
        
        # 2. Separation test
        logger.info("  Testing for separation...")
        sep_report = SeparationTest.test(self.X_train, self.y_train)
        diags["separation"] = sep_report
        if sep_report["separated"]:
            logger.warning(f"    {sep_report['warning']}")
        else:
            logger.info("    No separation detected.")
        
        # 3. Box-Tidwell test for linearity
        logger.info("  Testing linearity of logit (Box-Tidwell)...")
        bt_df = BoxTidwellTest.test(self.X_train, self.y_train, CONTINUOUS_FEATURES, alpha=0.05)
        if isinstance(bt_df, pd.DataFrame):
            n_nonlinear = (~bt_df["linear"]).sum()
            diags["box_tidwell"] = bt_df.to_dict("records")
            logger.info(f"    {n_nonlinear} of {len(bt_df)} features show non-linearity (p < 0.05)")
        else:
            diags["box_tidwell"] = bt_df
        
        self.diagnostics = diags
        return diags

    def fit(self):
        """Fit logistic regression using statsmodels."""
        logger.info("Fitting logistic regression...")
        
        # Prepare data: add constant for intercept
        X_train = sm.add_constant(self.X_train)
        
        # Fit model
        self.model = sm.Logit(self.y_train, X_train)
        self.result = self.model.fit(disp=0)
        
        logger.info(f"Model converged: {self.result.mle_retvals['converged']}")
        logger.info(f"McFadden R²: {self.result.prsquared:.4f}")
        
        return self.result.summary()

    def evaluate_and_threshold(self):
        """Evaluate on validation set and find optimal threshold."""
        logger.info("Evaluating on validation set...")
        
        # Predictions
        X_val_const = sm.add_constant(self.X_val)
        val_probs = self.result.predict(X_val_const)
        val_auc = roc_auc_score(self.y_val, val_probs)
        
        # Find optimal threshold (maximize Youden's J)
        fpr, tpr, thresholds = roc_curve(self.y_val, val_probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_thr = thresholds[optimal_idx]
        
        self.threshold = optimal_thr
        
        # Confusion matrix at optimal threshold
        val_preds = (val_probs >= optimal_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_val, val_preds).ravel()
        
        self.metrics = {
            "auc": float(val_auc),
            "auc_hgb_baseline": HGB_AUC,
            "auc_delta": float(val_auc - HGB_AUC),
            "optimal_threshold": float(optimal_thr),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
            "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan,
        }
        
        logger.info(f"  Validation AUC: {val_auc:.4f} (HGB: {HGB_AUC:.4f}, delta: {val_auc - HGB_AUC:+.4f})")
        logger.info(f"  Optimal threshold: {optimal_thr:.3f}")
        logger.info(f"  Sensitivity: {self.metrics['sensitivity']:.3f}, Specificity: {self.metrics['specificity']:.3f}")
        
        return self.metrics

    def extract_coefficients(self):
        """Extract and format coefficients with CIs and odds ratios.
        
        Returns:
            DataFrame with columns: feature, coef, se, p_value, ci_lower, ci_upper,
                                   odds_ratio, or_ci_lower, or_ci_upper
        """
        params = self.result.params[1:]  # Drop intercept
        pvalues = self.result.pvalues[1:]
        conf_int = self.result.conf_int().loc[params.index]
        conf_int.columns = ["ci_lower", "ci_upper"]
        
        coef_df = pd.DataFrame({
            "feature": params.index,
            "coef": params.values,
            "se": self.result.bse[1:].values,
            "p_value": pvalues.values,
            "ci_lower": conf_int["ci_lower"].values,
            "ci_upper": conf_int["ci_upper"].values,
        })
        
        # Odds ratios
        coef_df["odds_ratio"] = np.exp(coef_df["coef"])
        coef_df["or_ci_lower"] = np.exp(coef_df["ci_lower"])
        coef_df["or_ci_upper"] = np.exp(coef_df["ci_upper"])
        
        # Sort by absolute coefficient size
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).drop("abs_coef", axis=1)
        
        return coef_df

    def build_interpretation_table(self, coef_df):
        """Build plain-English interpretation of coefficients.
        
        Args:
            coef_df: Output from extract_coefficients().
        
        Returns:
            DataFrame with: feature, direction, effect_description, interpretation
        """
        interpretations = []
        
        for _, row in coef_df.iterrows():
            feature = row["feature"]
            coef = row["coef"]
            or_val = row["odds_ratio"]
            p_val = row["p_value"]
            ci_lower = row["or_ci_lower"]
            ci_upper = row["or_ci_upper"]
            
            # Direction
            direction = "+" if coef > 0 else "−"
            
            # Effect size (for interpretation)
            pct_change = (or_val - 1) * 100
            
            # Business interpretation (generic; can be customized per feature)
            if "user_total_orders" in feature:
                interpretation = (
                    f"Active customers are more likely to reorder. "
                    f"Each additional order in history increases odds by {abs(pct_change):.1f}%."
                )
            elif "user_reorder_ratio" in feature:
                interpretation = (
                    f"Repeat customers reorder more. "
                    f"A +1 point increase in reorder ratio increases odds by {abs(pct_change):.1f}%."
                )
            elif "up_order_count" in feature:
                interpretation = (
                    f"Purchase history is predictive. "
                    f"Each prior purchase increases reorder odds by {abs(pct_change):.1f}%."
                )
            elif "up_orders_since_last" in feature:
                interpretation = (
                    f"Recency matters. "
                    f"Each order since last purchase decreases odds by {abs(pct_change):.1f}%."
                )
            elif "product_reorder_rate" in feature:
                interpretation = (
                    f"Product-wide reorder propensity is predictive. "
                    f"A +1 point increase increases odds by {abs(pct_change):.1f}%."
                )
            elif "department_id" in feature or "aisle_id" in feature:
                interpretation = f"Category effects vary; reference level excluded."
            else:
                interpretation = (
                    f"A 1-unit increase in {feature} changes reorder odds by {pct_change:+.1f}%."
                )
            
            interpretations.append({
                "feature": feature,
                "direction": direction,
                "odds_ratio": float(or_val),
                "or_95_ci": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
                "p_value": float(p_val),
                "significant": "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns",
                "interpretation": interpretation,
            })
        
        return pd.DataFrame(interpretations)

    def to_report_dict(self):
        """Serialize model to report dictionary."""
        coef_df = self.extract_coefficients()
        interp_df = self.build_interpretation_table(coef_df)
        
        report = {
            "model": "Logistic Regression (statsmodels.api.Logit)",
            "data_info": {
                "sample_size": self.sample_size,
                "random_seed": self.random_seed,
                "features": FEATURES,
                "n_features": len(FEATURES),
            },
            "coefficients": coef_df[[
                "feature", "coef", "se", "p_value", "ci_lower", "ci_upper",
                "odds_ratio", "or_ci_lower", "or_ci_upper"
            ]].astype({
                "coef": float, "se": float, "p_value": float,
                "ci_lower": float, "ci_upper": float,
                "odds_ratio": float, "or_ci_lower": float, "or_ci_upper": float,
            }).to_dict("records"),
            "model_fit": {
                "n_obs": int(self.result.nobs),
                "mcfadden_r2": float(self.result.prsquared),
                "adj_mcfadden_r2": float(self.result.prsquared_adj),
                "log_likelihood": float(self.result.llf),
                "ll_null": float(self.result.llnull),
                "aic": float(self.result.aic),
                "bic": float(self.result.bic),
            },
            "validation_metrics": self.metrics,
            "interpretation": interp_df.to_dict("records"),
            "diagnostics": {
                "vif": self.diagnostics.get("vif", {}),
                "separation": self.diagnostics.get("separation", {}),
                "box_tidwell": (
                    self.diagnostics.get("box_tidwell", {})
                    if isinstance(self.diagnostics.get("box_tidwell"), dict)
                    else (self.diagnostics.get("box_tidwell", pd.DataFrame()).to_dict("records") if isinstance(self.diagnostics.get("box_tidwell"), pd.DataFrame) else [])
                ),
            },
        }
        
        return report


def main(sample_size=None):
    """Train logistic regression model and generate report."""
    import json
    from pathlib import Path
    
    # Setup
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    
    # Initialize and run
    model = LogisticRegressionModel(sample_size=sample_size, random_seed=42)
    
    logger.info("=" * 80)
    logger.info("MODEL 1: LOGISTIC REGRESSION ON REORDER PROBABILITY")
    logger.info("=" * 80)
    
    split_info = model.load_and_split()
    logger.info(f"Data split: {split_info}")
    
    model.run_diagnostics()
    
    model.fit()
    model.evaluate_and_threshold()
    
    report = model.to_report_dict()
    
    # Write report
    report_file = REPORT_PATH / "model_1_logit.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nSaved report: {report_file}")
    
    # Write coefficients to CSV
    coef_df = model.extract_coefficients()
    coef_file = REPORT_PATH / "model_1_coefficients.csv"
    coef_df.to_csv(coef_file, index=False)
    logger.info(f"Saved coefficients: {coef_file}")
    
    # Write interpretation to CSV
    interp_df = model.build_interpretation_table(coef_df)
    interp_file = REPORT_PATH / "model_1_interpretation.csv"
    interp_df.to_csv(interp_file, index=False)
    logger.info(f"Saved interpretation: {interp_file}")
    
    return model, report


if __name__ == "__main__":
    main()
