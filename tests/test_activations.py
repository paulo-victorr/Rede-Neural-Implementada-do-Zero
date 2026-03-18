"""Testes unitários para activations.py."""

import numpy as np
import pytest
from src.activations import relu, relu_derivada, sigmoid, sigmoid_derivada, softmax


class TestRelu:
    def test_valores_positivos_passam(self):
        z = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(relu(z), z)

    def test_valores_negativos_viram_zero(self):
        z = np.array([[-1.0, -5.0, 0.0]])
        esperado = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(relu(z), esperado)

    def test_mistura_de_valores(self):
        z = np.array([[-2.0, 0.0, 3.0]])
        esperado = np.array([[0.0, 0.0, 3.0]])
        np.testing.assert_array_equal(relu(z), esperado)

    def test_derivada_positivos(self):
        z = np.array([[1.0, 2.0]])
        np.testing.assert_array_equal(relu_derivada(z), np.array([[1.0, 1.0]]))

    def test_derivada_negativos(self):
        z = np.array([[-1.0, -2.0]])
        np.testing.assert_array_equal(relu_derivada(z), np.array([[0.0, 0.0]]))

    def test_derivada_zero(self):
        # Convenção: derivada em 0 é 0
        z = np.array([[0.0]])
        np.testing.assert_array_equal(relu_derivada(z), np.array([[0.0]]))


class TestSigmoid:
    def test_zero_retorna_meio(self):
        z = np.array([[0.0]])
        np.testing.assert_almost_equal(sigmoid(z), np.array([[0.5]]))

    def test_valor_grande_converge_para_um(self):
        z = np.array([[100.0]])
        assert sigmoid(z)[0, 0] > 0.9999

    def test_valor_pequeno_converge_para_zero(self):
        z = np.array([[-100.0]])
        assert sigmoid(z)[0, 0] < 0.0001

    def test_saida_entre_zero_e_um(self):
        z = np.random.randn(10, 5)
        s = sigmoid(z)
        assert np.all(s > 0) and np.all(s < 1)

    def test_derivada_em_zero(self):
        # sigmoid'(0) = 0.5 * 0.5 = 0.25
        z = np.array([[0.0]])
        np.testing.assert_almost_equal(sigmoid_derivada(z), np.array([[0.25]]))

    def test_derivada_sempre_positiva(self):
        z = np.random.randn(10, 5)
        assert np.all(sigmoid_derivada(z) > 0)


class TestSoftmax:
    def test_saida_soma_um_por_linha(self):
        z = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        somas = softmax(z).sum(axis=1)
        np.testing.assert_almost_equal(somas, np.ones(2))

    def test_saida_entre_zero_e_um(self):
        z = np.random.randn(5, 4)
        s = softmax(z)
        assert np.all(s > 0) and np.all(s < 1)

    def test_maior_logit_tem_maior_probabilidade(self):
        z = np.array([[1.0, 5.0, 2.0]])
        s = softmax(z)
        assert np.argmax(s) == 1

    def test_estabilidade_numerica_com_valores_grandes(self):
        z = np.array([[1000.0, 1001.0, 1002.0]])
        s = softmax(z)
        assert not np.any(np.isnan(s))
        np.testing.assert_almost_equal(s.sum(), 1.0)
