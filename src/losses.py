"""Funções de custo e suas derivadas."""

import numpy as np
from numpy.typing import NDArray


def mse(y_pred: NDArray, y_true: NDArray) -> float:
    """Erro quadrático médio entre predição e rótulo."""
    return float(np.mean((y_pred - y_true) ** 2))


def mse_derivada(y_pred: NDArray, y_true: NDArray) -> NDArray:
    """Gradiente do MSE em relação a y_pred."""
    return 2.0 * (y_pred - y_true) / y_true.size


def cross_entropy(y_pred: NDArray, y_true: NDArray, eps: float = 1e-12) -> float:
    """Cross-entropy categórica; y_true deve ser one-hot."""
    y_pred_clip = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.sum(y_true * np.log(y_pred_clip)) / y_true.shape[0])


def cross_entropy_derivada(y_pred: NDArray, y_true: NDArray) -> NDArray:
    """Gradiente combinado Softmax + Cross-entropy: ŷ - y (por amostra)."""
    return (y_pred - y_true) / y_true.shape[0]
