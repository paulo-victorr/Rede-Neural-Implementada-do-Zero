"""Testes unitários para layers.py."""

import numpy as np
import pytest
from src.layers import Layer
from src.activations import relu, relu_derivada, sigmoid, sigmoid_derivada, softmax


class TestLayerForward:
    def test_formato_saida(self):
        camada = Layer(4, 3, relu, relu_derivada)
        entrada = np.random.randn(5, 4)
        saida = camada.forward(entrada)
        assert saida.shape == (5, 3)

    def test_saida_relu_nao_negativa(self):
        camada = Layer(4, 8, relu, relu_derivada)
        entrada = np.random.randn(10, 4)
        saida = camada.forward(entrada)
        assert np.all(saida >= 0)

    def test_saida_sigmoid_entre_zero_e_um(self):
        camada = Layer(3, 5, sigmoid, sigmoid_derivada)
        entrada = np.random.randn(7, 3)
        saida = camada.forward(entrada)
        assert np.all(saida > 0) and np.all(saida < 1)

    def test_saida_softmax_soma_um(self):
        camada = Layer(4, 3, softmax, ativacao_derivada=None)
        entrada = np.random.randn(6, 4)
        saida = camada.forward(entrada)
        np.testing.assert_almost_equal(saida.sum(axis=1), np.ones(6))

    def test_armazena_entrada_e_z(self):
        camada = Layer(3, 2, relu, relu_derivada)
        entrada = np.random.randn(4, 3)
        camada.forward(entrada)
        assert camada._entrada is not None
        assert camada._z is not None
        assert camada._entrada.shape == (4, 3)
        assert camada._z.shape == (4, 2)

    def test_pesos_inicializados_he(self):
        # Variância esperada ≈ 2/n_entradas; verificamos que não são todos zero
        camada = Layer(100, 50, relu, relu_derivada)
        assert not np.all(camada.W == 0)
        assert camada.b.shape == (1, 50)


class TestLayerBackward:
    def test_formato_grad_propagado(self):
        camada = Layer(4, 3, relu, relu_derivada)
        entrada = np.random.randn(5, 4)
        camada.forward(entrada)
        grad_saida = np.random.randn(5, 3)
        grad_entrada = camada.backward(grad_saida)
        assert grad_entrada.shape == (5, 4)

    def test_formato_grad_W_e_b(self):
        camada = Layer(4, 3, relu, relu_derivada)
        entrada = np.random.randn(5, 4)
        camada.forward(entrada)
        camada.backward(np.random.randn(5, 3))
        assert camada.grad_W.shape == (4, 3)
        assert camada.grad_b.shape == (1, 3)

    def test_backward_sem_derivada_ativacao(self):
        # Caso softmax + cross-entropy: grad_saida já é delta final
        camada = Layer(4, 3, softmax, ativacao_derivada=None)
        entrada = np.random.randn(5, 4)
        camada.forward(entrada)
        grad_saida = np.random.randn(5, 3)
        grad_entrada = camada.backward(grad_saida)
        assert grad_entrada.shape == (5, 4)

    def test_gradiente_numerico_W(self):
        """Verifica grad_W com gradiente numérico (diferença finita)."""
        np.random.seed(0)
        camada = Layer(3, 2, sigmoid, sigmoid_derivada)
        entrada = np.random.randn(4, 3)
        y_true = np.random.randn(4, 2)

        def custo(W_flat):
            camada.W = W_flat.reshape(3, 2)
            pred = camada.forward(entrada)
            return float(np.mean((pred - y_true) ** 2))

        # Gradiente analítico
        camada.W = np.random.randn(3, 2)
        pred = camada.forward(entrada)
        grad_saida = 2.0 * (pred - y_true) / y_true.size
        camada.backward(grad_saida)
        grad_analitico = camada.grad_W.copy()

        # Gradiente numérico
        eps = 1e-5
        grad_numerico = np.zeros_like(camada.W)
        W_base = camada.W.copy()
        for i in range(W_base.shape[0]):
            for j in range(W_base.shape[1]):
                camada.W = W_base.copy()
                camada.W[i, j] += eps
                f_mais = custo(camada.W)
                camada.W = W_base.copy()
                camada.W[i, j] -= eps
                f_menos = custo(camada.W)
                grad_numerico[i, j] = (f_mais - f_menos) / (2 * eps)

        np.testing.assert_almost_equal(grad_analitico, grad_numerico, decimal=5)


class TestLayerAtualizarPesos:
    def test_pesos_mudam_apos_atualizacao(self):
        camada = Layer(3, 2, relu, relu_derivada)
        W_antes = camada.W.copy()
        b_antes = camada.b.copy()
        camada.forward(np.random.randn(4, 3))
        camada.backward(np.random.randn(4, 2))
        camada.atualizar_pesos(0.01)
        assert not np.allclose(camada.W, W_antes)
        assert not np.allclose(camada.b, b_antes)

    def test_taxa_aprendizado_zero_nao_muda_pesos(self):
        camada = Layer(3, 2, relu, relu_derivada)
        camada.forward(np.random.randn(4, 3))
        camada.backward(np.random.randn(4, 2))
        W_antes = camada.W.copy()
        b_antes = camada.b.copy()
        camada.atualizar_pesos(0.0)
        np.testing.assert_array_equal(camada.W, W_antes)
        np.testing.assert_array_equal(camada.b, b_antes)
