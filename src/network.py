"""Rede neural densa: orquestra camadas, treinamento e predição."""

from typing import Callable
import numpy as np
from numpy.typing import NDArray

from src.layers import Layer


class Network:
    """MLP composta por uma sequência de camadas densas."""

    def __init__(self, camadas: list[Layer]) -> None:
        """Recebe a lista de camadas já instanciadas, na ordem de forward."""
        self.camadas = camadas

    def _forward(self, X: NDArray) -> NDArray:
        """Passa X por todas as camadas em sequência."""
        saida = X
        for camada in self.camadas:
            saida = camada.forward(saida)
        return saida

    def _backward(self, grad: NDArray) -> None:
        """Propaga o gradiente da loss por todas as camadas em ordem inversa."""
        for camada in reversed(self.camadas):
            grad = camada.backward(grad)

    def _atualizar_pesos(self, taxa_aprendizado: float) -> None:
        """Atualiza pesos de todas as camadas."""
        for camada in self.camadas:
            camada.atualizar_pesos(taxa_aprendizado)

    def predict(self, X: NDArray) -> NDArray:
        """Retorna as probabilidades (ou saídas) para cada amostra de X."""
        return self._forward(X)

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        epochs: int,
        taxa_aprendizado: float,
        funcao_custo: Callable[[NDArray, NDArray], float],
        funcao_custo_derivada: Callable[[NDArray, NDArray], NDArray],
        tamanho_lote: int = 32,
        verbose: bool = True,
    ) -> list[float]:
        """Treina a rede e retorna o histórico de loss por época."""
        historico: list[float] = []
        n = X.shape[0]

        for epoca in range(1, epochs + 1):
            # Embaralha os dados a cada época
            indices = np.random.permutation(n)
            X_emb, y_emb = X[indices], y[indices]

            loss_epoca = 0.0
            n_lotes = 0

            for inicio in range(0, n, tamanho_lote):
                X_lote = X_emb[inicio : inicio + tamanho_lote]
                y_lote = y_emb[inicio : inicio + tamanho_lote]

                # Forward
                y_pred = self._forward(X_lote)

                # Loss do lote
                loss_epoca += funcao_custo(y_pred, y_lote)
                n_lotes += 1

                # Backward
                grad = funcao_custo_derivada(y_pred, y_lote)
                self._backward(grad)

                # Atualização
                self._atualizar_pesos(taxa_aprendizado)

            loss_media = loss_epoca / n_lotes
            historico.append(loss_media)

            if verbose and (epoca % max(1, epochs // 10) == 0 or epoca == 1):
                print(f"Época {epoca:>4}/{epochs} — loss: {loss_media:.6f}")

        return historico

    def evaluate(
        self,
        X: NDArray,
        y: NDArray,
        funcao_custo: Callable[[NDArray, NDArray], float],
    ) -> tuple[float, float]:
        """Retorna (loss, acurácia) no conjunto fornecido."""
        y_pred = self.predict(X)
        loss = funcao_custo(y_pred, y)
        acertos = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        acuracia = float(acertos / X.shape[0])
        return loss, acuracia
