"""Testes unitários para utils.py."""

import numpy as np
import pytest
from src.utils import one_hot, normalizar, dividir


class TestOneHot:
    def test_formato_saida(self):
        y = np.array([0, 1, 2])
        resultado = one_hot(y, n_classes=3)
        assert resultado.shape == (3, 3)

    def test_valores_corretos(self):
        y = np.array([0, 2, 1])
        esperado = np.array([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        np.testing.assert_array_equal(one_hot(y, n_classes=3), esperado)

    def test_cada_linha_soma_um(self):
        y = np.array([0, 1, 2, 0, 2])
        resultado = one_hot(y, n_classes=3)
        np.testing.assert_array_equal(resultado.sum(axis=1), np.ones(5))

    def test_n_classes_maior_que_labels(self):
        y = np.array([0, 1])
        resultado = one_hot(y, n_classes=5)
        assert resultado.shape == (2, 5)

    def test_aceita_floats_como_labels(self):
        y = np.array([0.0, 1.0, 2.0])
        resultado = one_hot(y, n_classes=3)
        assert resultado.shape == (3, 3)


class TestNormalizar:
    def test_media_zero_apos_normalizacao(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        X_norm = normalizar(X)
        np.testing.assert_almost_equal(X_norm.mean(axis=0), np.zeros(2))

    def test_desvio_um_apos_normalizacao(self):
        X = np.random.randn(50, 4) * 5 + 10
        X_norm = normalizar(X)
        np.testing.assert_almost_equal(X_norm.std(axis=0), np.ones(4), decimal=5)

    def test_formato_preservado(self):
        X = np.random.randn(20, 6)
        assert normalizar(X).shape == X.shape

    def test_feature_constante_nao_explode(self):
        X = np.ones((5, 3))
        resultado = normalizar(X)
        assert not np.any(np.isnan(resultado))
        assert not np.any(np.isinf(resultado))


class TestDividir:
    def test_proporcao_correta(self):
        X = np.arange(100).reshape(100, 1).astype(float)
        y = np.arange(100).astype(float)
        X_tr, X_val, y_tr, y_val = dividir(X, y, proporcao_treino=0.8)
        assert X_tr.shape[0] == 80
        assert X_val.shape[0] == 20

    def test_sem_sobreposicao(self):
        X = np.arange(50).reshape(50, 1).astype(float)
        y = np.arange(50).astype(float)
        X_tr, X_val, _, _ = dividir(X, y, proporcao_treino=0.7, semente=0)
        treino_vals = set(X_tr.flatten().tolist())
        val_vals = set(X_val.flatten().tolist())
        assert treino_vals.isdisjoint(val_vals)

    def test_todas_amostras_presentes(self):
        X = np.arange(40).reshape(40, 1).astype(float)
        y = np.arange(40).astype(float)
        X_tr, X_val, _, _ = dividir(X, y, proporcao_treino=0.75, semente=1)
        todos = set(X_tr.flatten().tolist()) | set(X_val.flatten().tolist())
        assert todos == set(range(40))

    def test_semente_garante_reproducibilidade(self):
        X = np.random.randn(60, 3)
        y = np.random.randn(60)
        X_tr1, X_val1, _, _ = dividir(X, y, semente=7)
        X_tr2, X_val2, _, _ = dividir(X, y, semente=7)
        np.testing.assert_array_equal(X_tr1, X_tr2)
        np.testing.assert_array_equal(X_val1, X_val2)

    def test_formatos_y_preservados(self):
        X = np.random.randn(30, 4)
        y = np.random.randn(30, 3)  # y pode ser one-hot
        _, _, y_tr, y_val = dividir(X, y, proporcao_treino=0.8)
        assert y_tr.shape[1] == 3
        assert y_val.shape[1] == 3
