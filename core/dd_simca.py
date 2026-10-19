"""DD-SIMCA-style one-class PCA modeling using moment-matched chi-square distances.

Independent Python implementation informed by Zontov et al. (2017), not a
line-by-line port. Calibration diagnostics are in-sample; use external sets
for generalization claims.
"""
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA


def _moments(z, name):
    m = float(np.mean(z))
    v = float(np.var(z, ddof=1))
    if m <= 1e-14 or v <= 1e-14:
        raise ValueError(f'Variabilidade insuficiente para estimar a distância {name}.')
    return 2 * m * m / v, m


class DDSIMCA:
    def __init__(self, n_components=2, alpha=0.05, gamma=0.01, scale=False):
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.scale = bool(scale)

    def fit(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
            raise ValueError('X deve ser uma matriz numérica 2D, sem valores ausentes.')
        n, p = x.shape
        if n < 8 or not 1 <= self.n_components < min(n, p):
            raise ValueError('São necessárias ≥8 amostras da classe-alvo e 1 ≤ PCs < min(amostras, variáveis).')
        if not 0 < self.alpha < 1 or not 0 < self.gamma < 1:
            raise ValueError('Alpha e gamma devem estar entre zero e um.')
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0, ddof=1) if self.scale else np.ones(p)
        if np.any(self.std_ < 1e-12):
            raise ValueError('Há variáveis constantes. Exclua-as ou desative o autoescalamento.')
        z = (x - self.mean_) / self.std_
        self.pca_ = PCA(n_components=self.n_components, svd_solver='full').fit(z)
        self.eigenvalues_ = self.pca_.explained_variance_
        self.n_training_ = n
        sd, od = self._distances(x)
        self.df_sd_, self.sd_mean_ = _moments(sd, 'SD')
        self.df_od_, self.od_mean_ = _moments(od, 'OD')
        self.df_total_ = self.df_sd_ + self.df_od_
        # Gamma is a model-wise outlier probability; per-object threshold.
        self.acceptance_ = float(chi2.ppf(1-self.alpha, self.df_total_))
        self.outlier_ = float(chi2.ppf((1-self.gamma)**(1/n), self.df_total_))
        if self.outlier_ < self.acceptance_:
            self.outlier_ = self.acceptance_
        return self

    def _distances(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim == 1: x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(self.mean_) or not np.isfinite(x).all():
            raise ValueError('X deve ter as mesmas variáveis do treinamento, sem NaN/inf.')
        z = (x - self.mean_) / self.std_
        t = self.pca_.transform(z)
        reconstructed = self.pca_.inverse_transform(t)
        sd = np.sum(t*t / self.eigenvalues_, axis=1)
        od = np.sum((z-reconstructed)**2, axis=1)
        return sd, od

    def predict_detail(self, X):
        sd, od = self._distances(X)
        distance = self.df_sd_ * sd / self.sd_mean_ + self.df_od_ * od / self.od_mean_
        accepted = distance <= self.acceptance_
        extreme = (distance > self.acceptance_) & (distance <= self.outlier_)
        outlier = distance > self.outlier_
        return {'SD':sd, 'OD':od, 'distance':distance, 'accepted':accepted,
                'extreme':extreme, 'outlier':outlier,
                'status':np.where(outlier,'Outlier',np.where(extreme,'Extrema','Aceita'))}

    def predict(self, X):
        return self.predict_detail(X)['accepted']

    def acceptance_boundary(self, n=200, outlier=False):
        threshold = self.outlier_ if outlier else self.acceptance_
        sd = np.linspace(0, threshold*self.sd_mean_/self.df_sd_, n)
        od = np.maximum(0, (threshold-self.df_sd_*sd/self.sd_mean_)*self.od_mean_/self.df_od_)
        return sd, od
