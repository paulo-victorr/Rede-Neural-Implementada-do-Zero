"""Camada densa com forward e backward pass."""

from typing import Callable
import numpy as np
from numpy.typing import NDArray


class Layer:
    """Camada totalmente conectada (dense) com uma função de ativação."""

    def __init__(
        self,
        n_entradas: int,
        n_neuronios: int,
        ativacao: Callable[[NDArray], NDArray],
        ativacao_derivada: Callable[[NDArray], NDArray] | None = None,
    ) -> None:
        """Inicializa pesos (He), biases e função de ativação."""
        # Inicialização He: boa para ReLU e variantes
        self.W: NDArray = np.random.randn(n_entradas, n_neuronios) * np.sqrt(2.0 / n_entradas)
        self.b: NDArray = np.zeros((1, n_neuronios))
        self.ativacao = ativacao
        self.ativacao_derivada = ativacao_derivada

        # Valores armazenados no forward para uso no backward
        self._entrada: NDArray | None = None
        self._z: NDArray | None = None

        # Gradientes calculados no backward
        self.grad_W: NDArray | None = None
        self.grad_b: NDArray | None = None

    def forward(self, entrada: NDArray) -> NDArray:
        """Calcula a saída da camada: ativacao(entrada @ W + b)."""
        self._entrada = entrada
        self._z = entrada @ self.W + self.b
        return self.ativacao(self._z)

    def backward(self, grad_saida: NDArray) -> NDArray:
        """Calcula gradientes e propaga o erro para a camada anterior.

        Para camadas com softmax na saída, grad_saida já deve ser o gradiente
        combinado (ŷ - y), então ativacao_derivada pode ser None.
        """
        n = self._entrada.shape[0]

        # Aplica derivada da ativação, se necessário
        if self.ativacao_derivada is not None:
            delta = grad_saida * self.ativacao_derivada(self._z)
        else:
            # Gradiente já vem combinado (ex: softmax + cross-entropy)
            delta = grad_saida

        self.grad_W = self._entrada.T @ delta
        self.grad_b = np.sum(delta, axis=0, keepdims=True)

        # Gradiente a propagar para a camada anterior
        return delta @ self.W.T

    def atualizar_pesos(self, taxa_aprendizado: float) -> None:
        """Atualiza W e b com gradiente descendente simples."""
        self.W -= taxa_aprendizado * self.grad_W
        self.b -= taxa_aprendizado * self.grad_b
