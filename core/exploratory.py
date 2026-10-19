"""Spectral preprocessing and exploratory PCA for Marvin Quimiometria.

All fit-dependent preprocessing is fitted on a calibration set only.
For predictive cross-validation, re-fit it separately inside each fold.
"""
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


def preprocess(X, *, snv=False, msc=False, smooth=False, window=11, polyorder=2,
               derivative=0, baseline=False):
    """Return a processed copy of a complete, finite 2-D spectral array.

    MSC uses the dataset mean spectrum as reference, so processing a new data set
    requires saving/reusing the calibration reference (not supplied here).
    This is exploratory preprocessing; not a ready-made predictive pipeline.
    """
    A = np.asarray(X, dtype=float).copy()
    if A.ndim != 2 or min(A.shape) < 2 or not np.isfinite(A).all():
        raise ValueError('X deve ser uma matriz 2D, completa e finita, com ao menos 2 linhas e 2 colunas.')
    if baseline:
        axis = np.arange(A.shape[1], dtype=float)
        start, stop = A[:, 0], A[:, -1]
        A -= start[:, None] + (stop-start)[:, None] * (axis[None, :]/(A.shape[1]-1))
    if smooth or derivative:
        if int(window) != window or window < 3 or window % 2 != 1 or window > A.shape[1]:
            raise ValueError('A janela Savitzky–Golay deve ser ímpar, >=3 e <= número de variáveis.')
        if polyorder < 0 or polyorder >= window or derivative < 0 or derivative > polyorder:
            raise ValueError('É necessário 0 <= derivada <= ordem polinomial < janela.')
        A = savgol_filter(A, int(window), polyorder=int(polyorder), deriv=int(derivative), axis=1)
    if snv:
        s = A.std(axis=1, ddof=1)
        if np.any(s < 1e-12):
            raise ValueError('SNV não pode ser aplicado a espectros constantes.')
        A = (A-A.mean(axis=1, keepdims=True))/s[:, None]
    if msc:
        ref = A.mean(axis=0)
        m = ref.mean()
        denom = np.sum((ref-m)**2)
        if denom < 1e-20:
            raise ValueError('MSC exige espectro de referência com variação.')
        slopes = ((A-A.mean(axis=1, keepdims=True))@(ref-m))/denom
        intercepts = A.mean(axis=1)-slopes*m
        if np.any(np.abs(slopes) < 1e-12):
            raise ValueError('MSC encontrou inclinação praticamente nula.')
        A = (A-intercepts[:, None])/slopes[:, None]
    return A


def exploratory_pca(X, n_components=2, autoscale=False):
    A = np.asarray(X, dtype=float)
    if A.ndim != 2 or not np.isfinite(A).all() or len(A) < 3:
        raise ValueError('PCA exige pelo menos três amostras e matriz numérica sem ausentes.')
    mean = A.mean(axis=0)
    std = A.std(axis=0, ddof=1)
    if autoscale and np.any(std < 1e-12):
        raise ValueError('Remova variáveis constantes antes de autoescalar.')
    scale = std if autoscale else np.ones(A.shape[1])
    Z=(A-mean)/scale
    max_components=min(Z.shape[0]-1,Z.shape[1],np.linalg.matrix_rank(Z))
    if n_components < 1 or n_components > max_components:
        raise ValueError(f'Número de componentes deve estar entre 1 e {max_components}.')
    model=PCA(n_components=n_components, svd_solver='full')
    scores=model.fit_transform(Z)
    reconstruction=model.inverse_transform(scores)
    residual=Z-reconstruction
    q=np.sum(residual**2,axis=1)
    eigenvalues=model.explained_variance_
    t2=np.sum(scores**2/eigenvalues,axis=1)
    return dict(scores=scores,loadings=model.components_.T,
                explained=model.explained_variance_ratio_,
                q=q,t2=t2,reconstructed=reconstruction*scale+mean,
                mean=mean,scale=scale)
