"""Regression analysis module for ReorderIQ.

Provides interpretable statistical models for:
  - Logistic regression on reorder probability (Model 1)
  - Gamma GLM or OLS on forecast error (Model 2)
  - Poisson/Negative Binomial on order frequency (Model 3)

All models use statsmodels for p-values, standard errors, and confidence intervals.
Designed for MBA Business Analytics portfolio: statistical rigor and interpretability
over raw performance.
"""

__version__ = "1.0"
