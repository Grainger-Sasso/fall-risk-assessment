"""Shared preprocessing for the classifier pipelines."""

import numpy as np
from sklearn.preprocessing import RobustScaler

# Iteration cap for libsvm fits. A convergent fit at this cohort size needs
# ~10^3 iterations, so the cap never changes a converged solution; it only
# stops a non-convergent one (which would otherwise spin forever, since
# sklearn's default max_iter=-1 is unlimited) with a ConvergenceWarning.
SVM_MAX_ITER = 1_000_000


class DegenerateSafeRobustScaler(RobustScaler):
    """RobustScaler that does not explode on point-mass-plus-outlier columns.

    RobustScaler divides by each column's IQR and only treats an IQR below
    ~10 machine epsilons as zero. A quantized feature whose values almost all
    coincide (e.g. a within-participant IQR that is 0.02 for nearly every
    participant) has a cross-participant IQR of floating-point noise
    (~1e-10) that passes that check, so a single outlier scales to ~1e9 and
    makes margin-based models (SVMs) non-convergent.

    A column is degenerate when its IQR is negligible relative to its spread
    (``iqr < rel_tol * std``). Such columns fall back to standard-deviation
    scaling; well-behaved columns are scaled exactly as RobustScaler would.
    """

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
        rel_tol=1e-6,
    ):
        super().__init__(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy,
            unit_variance=unit_variance,
        )
        self.rel_tol = rel_tol

    def fit(self, X, y=None):
        super().fit(X, y)
        self.degenerate_columns_ = np.zeros(self.n_features_in_, dtype=bool)
        if not self.with_scaling:
            return self
        X_arr = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            std = np.nanstd(X_arr, axis=0)
        iqr = self.scale_  # after RobustScaler's own zero handling
        degenerate = (std > 0) & (iqr < self.rel_tol * std)
        if np.any(degenerate):
            scale = np.array(self.scale_, dtype=float, copy=True)
            scale[degenerate] = std[degenerate]
            self.scale_ = scale
            self.degenerate_columns_ = degenerate
        return self
