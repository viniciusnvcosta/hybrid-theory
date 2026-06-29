"""EVT/GPD residual thresholding for anomaly detection in residuals.

Uses Peaks-Over-Threshold method with Generalized Pareto Distribution.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from cdade.registry import register_reconciler


@register_reconciler("evt")
class EVTReconciler:
    """EVT/GPD residual thresholding for anomaly detection in residuals.

    Uses Peaks-Over-Threshold method with Generalized Pareto Distribution.

    Args:
        cfg: Configuration object
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.threshold = None
        self.shape = None
        self.scale = None

    def fit(self, residuals: pd.Series) -> "EVTReconciler":
        """Fit GPD to residuals above threshold.

        Args:
            residuals: Residual series to analyze

        Returns:
            Self for chaining
        """
        # Select top k residuals (e.g., 5%)
        n_threshold = int(self.cfg.contamination * len(residuals))
        self.threshold = residuals.abs().nlargest(n_threshold).min()

        # Fit GPD to exceedances
        exceedances = residuals[residuals > self.threshold] - self.threshold

        if len(exceedances) > 0:
            self.scale, self.shape = genpareto.fit(exceedances)

        return self

    def reconcile(
        self,
        leaf_forecasts: pd.DataFrame,
        residuals: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Apply EVT thresholding to residuals.

        Args:
            leaf_forecasts: Leaf-level forecasts
            residuals: Raw residuals

        Returns:
            Tuple of (reconciled_leaf_series, reconciled_aggregate_series, thresholded_residuals)
        """
        # Apply GPD threshold to residuals
        thresholded_residuals = np.where(
            residuals.abs() > self.threshold,
            residuals,
            0,
        )

        # Add thresholded residuals to reconciled forecasts
        reconciled_leaves = leaf_forecasts + thresholded_residuals
        reconciled_aggregate = reconciled_leaves.sum(axis=1)

        reconciled_leaf_series = reconciled_leaves
        reconciled_aggregate_series = pd.Series(
            reconciled_aggregate,
            name="aggregate",
        )

        return reconciled_leaf_series, reconciled_aggregate_series, thresholded_residuals
