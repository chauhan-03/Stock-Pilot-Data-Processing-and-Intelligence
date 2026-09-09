"""Diagnostic tests for regression models.

Provides:
  - VIF (Variance Inflation Factor) for multicollinearity
  - Box-Tidwell test for linearity of logit
  - Breusch-Pagan test for heteroscedasticity
  - Separation detection for logistic regression
  - Overdispersion tests for count models
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_rainbow_statistic

logger = logging.getLogger(__name__)


class VIFDiagnostic:
    """Compute and report Variance Inflation Factors."""

    @staticmethod
    def compute(X, feature_names=None):
        """Compute VIF for all numeric columns.
        
        Args:
            X: DataFrame with numeric features only.
            feature_names: Optional list of feature names. Defaults to X.columns.
        
        Returns:
            DataFrame with columns: feature, vif, flagged (bool, VIF > 5)
        """
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        vif_data = []
        for i, col in enumerate(feature_names):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_data.append({"feature": col, "vif": float(vif), "flagged": vif > 5})
            except Exception as e:
                logger.warning(f"VIF computation failed for {col}: {e}")
                vif_data.append({"feature": col, "vif": np.nan, "flagged": None})
        
        df = pd.DataFrame(vif_data)
        df = df.sort_values("vif", ascending=False, na_position="last")
        return df

    @staticmethod
    def report(vif_df):
        """Generate human-readable VIF report.
        
        Args:
            vif_df: Output from compute().
        
        Returns:
            dict with summary and list of violations.
        """
        max_vif = vif_df["vif"].max()
        violations = vif_df[vif_df["flagged"] == True][["feature", "vif"]].to_dict("records")
        
        return {
            "max_vif": float(max_vif),
            "n_violations": len(violations),
            "violations": violations,
            "pass": len(violations) == 0,
            "recommendation": (
                "All VIF < 5. No collinearity concerns."
                if len(violations) == 0
                else f"{len(violations)} features with VIF > 5. Consider removal or regularization."
            ),
        }


class SeparationTest:
    """Detect complete or quasi-complete separation in logistic regression."""

    @staticmethod
    def test(X, y):
        """Check for separation in binary classification data.
        
        Uses a simple heuristic: for each feature, check if positive and negative
        classes have non-overlapping ranges (or very narrow overlap).
        
        Args:
            X: DataFrame of features.
            y: Binary target (0/1).
        
        Returns:
            dict with 'separated' (bool), 'features' (list of problematic features),
            and 'severity' (str: 'none', 'potential', 'likely').
        """
        separated_features = []
        
        for col in X.select_dtypes(include=[np.number]).columns:
            x0 = X.loc[y == 0, col].dropna()
            x1 = X.loc[y == 1, col].dropna()
            
            if len(x0) == 0 or len(x1) == 0:
                continue
            
            # Check for non-overlapping ranges
            if x0.max() < x1.min() or x1.max() < x0.min():
                separated_features.append({"feature": col, "severity": "complete"})
            # Check for near-complete separation (overlap < 5% of range)
            elif max(x0.max(), x1.max()) - min(x0.min(), x1.min()) > 0:
                overlap_range = min(x0.max(), x1.max()) - max(x0.min(), x1.min())
                total_range = max(x0.max(), x1.max()) - min(x0.min(), x1.min())
                overlap_pct = overlap_range / total_range if total_range > 0 else 0
                if overlap_pct < 0.05:
                    separated_features.append({"feature": col, "severity": "quasi-complete"})
        
        if len(separated_features) > 0:
            severity = "likely" if any(f["severity"] == "complete" for f in separated_features) else "potential"
            return {
                "separated": True,
                "severity": severity,
                "features": separated_features,
                "warning": (
                    f"{severity.upper()}: {len(separated_features)} feature(s) show separation. "
                    "Model may not converge or coefficients may be unreliable. Consider removal or regularization."
                ),
            }
        else:
            return {
                "separated": False,
                "severity": "none",
                "features": [],
                "warning": "No separation detected.",
            }


class BoxTidwellTest:
    """Test linearity of the logit via Box-Tidwell transformation."""

    @staticmethod
    def test(X, y, continuous_features, alpha=0.05):
        """Perform Box-Tidwell test on continuous predictors.
        
        Box-Tidwell tests H0: log(x_i) term is not significant in logit(y) ~ x_i + log(x_i).
        Rejection suggests non-linearity in the logit.
        
        Args:
            X: DataFrame of features.
            y: Binary target (0/1).
            continuous_features: List of continuous feature names to test.
            alpha: Significance level (default 0.05).
        
        Returns:
            dict with results per feature: feature, t_stat, p_value, linear (bool).
        """
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            logger.warning("statsmodels not available; skipping Box-Tidwell")
            return {"error": "statsmodels not available", "results": []}
        
        results = []
        df = pd.DataFrame(X)
        df["y"] = y.values
        
        for feat in continuous_features:
            if feat not in df.columns:
                continue
            
            # Skip if feature has 0 or negative values (log undefined)
            if (df[feat] <= 0).any():
                results.append({
                    "feature": feat,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "linear": None,
                    "note": "Feature has non-positive values; log transform not applicable.",
                })
                continue
            
            try:
                # Fit model: y ~ x + log(x)
                df[f"{feat}_log"] = np.log(df[feat])
                formula = f"y ~ {feat} + {feat}_log"
                model = smf.logit(formula, data=df).fit(disp=0)
                
                # Extract t-stat and p-value for the log(x) term
                coef_name = f"{feat}_log"
                if coef_name in model.pvalues.index:
                    p_val = model.pvalues[coef_name]
                    t_stat = model.tvalues[coef_name]
                    results.append({
                        "feature": feat,
                        "t_stat": float(t_stat),
                        "p_value": float(p_val),
                        "linear": p_val > alpha,  # True if we *fail* to reject H0 (linearity holds)
                    })
                else:
                    results.append({
                        "feature": feat,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "linear": None,
                        "note": f"Could not extract {coef_name} from model.",
                    })
            except Exception as e:
                logger.warning(f"Box-Tidwell test failed for {feat}: {e}")
                results.append({
                    "feature": feat,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "linear": None,
                    "note": str(e),
                })
        
        return pd.DataFrame(results)


class OverdispersionTest:
    """Test for overdispersion in Poisson models."""

    @staticmethod
    def pearson_chi2_test(model_result):
        """Pearson chi-square test for overdispersion.
        
        Test H0: mean = variance (Poisson assumption).
        Statistic: sum((y - mu)^2 / mu) ~ chi2(df).
        
        Args:
            model_result: Fitted statsmodels GLM result (e.g., Poisson).
        
        Returns:
            dict with statistic, df, p_value, ratio (chi2/df), overdispersed (bool).
        """
        y = model_result.model.endog
        mu = model_result.fittedvalues
        
        # Pearson chi-square
        chi2_stat = np.sum((y - mu) ** 2 / mu)
        df = len(y) - model_result.params.shape[0]
        p_val = 1 - stats.chi2.cdf(chi2_stat, df)
        ratio = chi2_stat / df if df > 0 else np.nan
        
        return {
            "test": "Pearson chi-square",
            "statistic": float(chi2_stat),
            "df": int(df),
            "p_value": float(p_val),
            "ratio": float(ratio),
            "overdispersed": ratio > 1.5 if not np.isnan(ratio) else None,  # Overdispersion if ratio >> 1
            "interpretation": (
                "Poisson assumption holds (ratio ~1)." if ratio is np.nan or ratio < 1.2
                else f"Evidence of overdispersion (ratio={ratio:.2f}); Negative Binomial preferred."
            ),
        }

    @staticmethod
    def cameron_trivedi_test(model_result):
        """Cameron-Trivedi overdispersion test.
        
        Regresses (y - mu)^2 - y on mu. Significance of slope indicates overdispersion.
        
        Args:
            model_result: Fitted statsmodels GLM result (Poisson).
        
        Returns:
            dict with statistic, p_value, overdispersed (bool).
        """
        from scipy import stats as sp_stats
        
        y = model_result.model.endog
        mu = model_result.fittedvalues
        
        # Test variable: (y - mu)^2 - y
        test_var = (y - mu) ** 2 - y
        
        # Auxiliary regression: test_var ~ mu, extract t-stat
        design = np.column_stack([np.ones(len(mu)), mu])
        try:
            b = np.linalg.lstsq(design, test_var, rcond=None)[0]
            resid = test_var - design @ b
            sigma2 = np.sum(resid ** 2) / (len(y) - 2)
            se_b1 = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])
            t_stat = b[1] / se_b1
            p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), len(y) - 2))
            
            return {
                "test": "Cameron-Trivedi",
                "statistic": float(t_stat),
                "p_value": float(p_val),
                "overdispersed": p_val < 0.05,
                "interpretation": (
                    "No significant overdispersion (p >= 0.05)." if p_val >= 0.05
                    else "Significant overdispersion (p < 0.05); Negative Binomial preferred."
                ),
            }
        except Exception as e:
            logger.warning(f"Cameron-Trivedi test failed: {e}")
            return {"test": "Cameron-Trivedi", "error": str(e)}


class BreuschPaganTest:
    """Breusch-Pagan test for heteroscedasticity."""

    @staticmethod
    def test(residuals, X):
        """Perform Breusch-Pagan test.
        
        Args:
            residuals: Model residuals (e.g., Pearson residuals).
            X: Design matrix (features).
        
        Returns:
            dict with statistic, p_value, heteroscedastic (bool).
        """
        from statsmodels.stats.diagnostic import het_breuschpagan
        
        try:
            # het_breuschpagan returns (LM_stat, LM_pvalue, F_stat, F_pvalue)
            lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(residuals, X)
            return {
                "test": "Breusch-Pagan",
                "lm_statistic": float(lm_stat),
                "lm_p_value": float(lm_pval),
                "f_statistic": float(f_stat),
                "f_p_value": float(f_pval),
                "heteroscedastic": lm_pval < 0.05,
                "interpretation": (
                    "Homoscedasticity assumption holds (p >= 0.05)." if lm_pval >= 0.05
                    else "Evidence of heteroscedasticity (p < 0.05). Consider weighted regression."
                ),
            }
        except Exception as e:
            logger.warning(f"Breusch-Pagan test failed: {e}")
            return {"test": "Breusch-Pagan", "error": str(e)}
