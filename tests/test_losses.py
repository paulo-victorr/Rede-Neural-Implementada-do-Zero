"""Testes unitários para losses.py."""

import numpy as np
import pytest
from src.losses import mse, mse_derivada, cross_entropy, cross_entropy_derivada


class TestMse:
    def test_predicao_perfeita_retorna_zero(self):
        y = np.array([[1.0, 0.0, 0.0]])
        assert mse(y, y) == 0.0

    def test_valor_conhecido(self):
        y_pred = np.array([[2.0]])
        y_true = np.array([[0.0]])
        # (2-0)^2 / 1 = 4.0
        np.testing.assert_almost_equal(mse(y_pred, y_true), 4.0)

    def test_retorna_escalar(self):
        y = np.random.randn(8, 3)
        assert isinstance(mse(y, y), float)

    def test_sempre_nao_negativo(self):
        y_pred = np.random.randn(5, 4)
        y_true = np.random.randn(5, 4)
        assert mse(y_pred, y_true) >= 0.0

    def test_derivada_formato(self):
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = np.array([[0.0, 0.0], [0.0, 0.0]])
        assert mse_derivada(y_pred, y_true).shape == y_pred.shape

    def test_derivada_valor_conhecido(self):
        y_pred = np.array([[1.0]])
        y_true = np.array([[0.0]])
        # 2*(1-0)/1 = 2.0
        np.testing.assert_almost_equal(mse_derivada(y_pred, y_true), np.array([[2.0]]))

    def test_derivada_predicao_perfeita_e_zero(self):
        y = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(mse_derivada(y, y), np.zeros_like(y))


class TestCrossEntropy:
    def test_predicao_perfeita_retorna_zero(self):
        y_pred = np.array([[1.0, 0.0, 0.0]])
        y_true = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_almost_equal(cross_entropy(y_pred, y_true), 0.0, decimal=5)

    def test_sempre_nao_negativo(self):
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert cross_entropy(y_pred, y_true) >= 0.0

    def test_retorna_escalar(self):
        y_pred = np.array([[0.5, 0.3, 0.2]])
        y_true = np.array([[1.0, 0.0, 0.0]])
        assert isinstance(cross_entropy(y_pred, y_true), float)

    def test_confianca_errada_tem_custo_maior(self):
        y_true = np.array([[1.0, 0.0, 0.0]])
        y_certo = np.array([[0.9, 0.05, 0.05]])
        y_errado = np.array([[0.1, 0.8, 0.1]])
        assert cross_entropy(y_errado, y_true) > cross_entropy(y_certo, y_true)

    def test_nao_explode_com_pred_zero(self):
        y_pred = np.array([[0.0, 1.0]])
        y_true = np.array([[1.0, 0.0]])
        resultado = cross_entropy(y_pred, y_true)
        assert np.isfinite(resultado)

    def test_derivada_formato(self):
        y_pred = np.array([[0.7, 0.2, 0.1]])
        y_true = np.array([[1.0, 0.0, 0.0]])
        assert cross_entropy_derivada(y_pred, y_true).shape == y_pred.shape

    def test_derivada_gradiente_softmax_ce(self):
        # Gradiente combinado: (ŷ - y) / n_amostras
        y_pred = np.array([[0.7, 0.2, 0.1]])
        y_true = np.array([[1.0, 0.0, 0.0]])
        esperado = (y_pred - y_true) / 1
        np.testing.assert_almost_equal(cross_entropy_derivada(y_pred, y_true), esperado)

    def test_derivada_media_por_batch(self):
        y_pred = np.array([[0.6, 0.4], [0.3, 0.7]])
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        grad = cross_entropy_derivada(y_pred, y_true)
        esperado = (y_pred - y_true) / 2
        np.testing.assert_almost_equal(grad, esperado)
