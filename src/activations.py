"""Funções de ativação e suas derivadas."""

import numpy as np
from numpy.typing import NDArray


def relu(z: NDArray) -> NDArray:
    """Aplica ReLU elemento a elemento: max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivada(z: NDArray) -> NDArray:
    """Derivada da ReLU: 1 onde z > 0, 0 caso contrário."""
    return (z > 0).astype(float)


def sigmoid(z: NDArray) -> NDArray:
    """Aplica a função sigmoid: 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivada(z: NDArray) -> NDArray:
    """Derivada da sigmoid em função de z."""
    s = sigmoid(z)
    return s * (1.0 - s)


def softmax(z: NDArray) -> NDArray:
    """Aplica softmax linha a linha, com estabilidade numérica."""
    z_estavel = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_estavel)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
