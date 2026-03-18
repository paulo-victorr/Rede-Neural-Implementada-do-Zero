"""Funções utilitárias: codificação, normalização e divisão de dados."""

import numpy as np
from numpy.typing import NDArray


def one_hot(y: NDArray, n_classes: int) -> NDArray:
    """Converte vetor de rótulos inteiros em matriz one-hot."""
    return np.eye(n_classes)[y.astype(int)]


def normalizar(X: NDArray) -> NDArray:
    """Normaliza X para média 0 e desvio padrão 1 por feature (coluna)."""
    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    desvio[desvio == 0] = 1.0  # evita divisão por zero em features constantes
    return (X - media) / desvio


def dividir(
    X: NDArray,
    y: NDArray,
    proporcao_treino: float = 0.8,
    semente: int | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Divide (X, y) em treino e validação de forma embaralhada.

    Retorna (X_treino, X_val, y_treino, y_val).
    """
    if semente is not None:
        np.random.seed(semente)

    n = X.shape[0]
    indices = np.random.permutation(n)
    corte = int(n * proporcao_treino)

    idx_treino, idx_val = indices[:corte], indices[corte:]
    return X[idx_treino], X[idx_val], y[idx_treino], y[idx_val]
