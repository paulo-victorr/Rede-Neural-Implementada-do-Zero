"""Testes unitários para network.py."""

import numpy as np
import pytest
from src.layers import Layer
from src.network import Network
from src.activations import relu, relu_derivada, softmax
from src.losses import cross_entropy, cross_entropy_derivada, mse, mse_derivada


def _rede_simples() -> Network:
    """Rede 4 → 8 (ReLU) → 3 (Softmax) para testes."""
    return Network([
        Layer(4, 8, relu, relu_derivada),
        Layer(8, 3, softmax, ativacao_derivada=None),
    ])


def _dados_classificacao(n: int = 20) -> tuple:
    """Gera dados e rótulos one-hot aleatórios para 3 classes."""
    np.random.seed(42)
    X = np.random.randn(n, 4)
    classes = np.random.randint(0, 3, size=n)
    y = np.eye(3)[classes]
    return X, y


class TestNetworkPredict:
    def test_formato_saida(self):
        rede = _rede_simples()
        X, _ = _dados_classificacao()
        saida = rede.predict(X)
        assert saida.shape == (20, 3)

    def test_saida_softmax_soma_um(self):
        rede = _rede_simples()
        X, _ = _dados_classificacao()
        saida = rede.predict(X)
        np.testing.assert_almost_equal(saida.sum(axis=1), np.ones(20))

    def test_saida_entre_zero_e_um(self):
        rede = _rede_simples()
        X, _ = _dados_classificacao()
        saida = rede.predict(X)
        assert np.all(saida > 0) and np.all(saida < 1)


class TestNetworkFit:
    def test_retorna_historico_com_tamanho_certo(self):
        rede = _rede_simples()
        X, y = _dados_classificacao()
        hist = rede.fit(X, y, epochs=5, taxa_aprendizado=0.01,
                        funcao_custo=cross_entropy,
                        funcao_custo_derivada=cross_entropy_derivada,
                        verbose=False)
        assert len(hist) == 5

    def test_loss_diminui_ao_longo_do_treinamento(self):
        # Treina por mais épocas e verifica tendência de queda
        np.random.seed(0)
        rede = _rede_simples()
        X, y = _dados_classificacao(n=100)
        hist = rede.fit(X, y, epochs=200, taxa_aprendizado=0.05,
                        funcao_custo=cross_entropy,
                        funcao_custo_derivada=cross_entropy_derivada,
                        tamanho_lote=32, verbose=False)
        assert hist[-1] < hist[0]

    def test_loss_e_escalar_positivo(self):
        rede = _rede_simples()
        X, y = _dados_classificacao()
        hist = rede.fit(X, y, epochs=3, taxa_aprendizado=0.01,
                        funcao_custo=cross_entropy,
                        funcao_custo_derivada=cross_entropy_derivada,
                        verbose=False)
        assert all(v > 0 for v in hist)

    def test_pesos_mudam_apos_fit(self):
        rede = _rede_simples()
        W_antes = rede.camadas[0].W.copy()
        X, y = _dados_classificacao()
        rede.fit(X, y, epochs=1, taxa_aprendizado=0.01,
                 funcao_custo=cross_entropy,
                 funcao_custo_derivada=cross_entropy_derivada,
                 verbose=False)
        assert not np.allclose(rede.camadas[0].W, W_antes)

    def test_tamanho_lote_maior_que_dataset(self):
        # lote maior que dataset não deve quebrar
        rede = _rede_simples()
        X, y = _dados_classificacao(n=10)
        hist = rede.fit(X, y, epochs=2, taxa_aprendizado=0.01,
                        funcao_custo=cross_entropy,
                        funcao_custo_derivada=cross_entropy_derivada,
                        tamanho_lote=100, verbose=False)
        assert len(hist) == 2


class TestNetworkEvaluate:
    def test_retorna_loss_e_acuracia(self):
        rede = _rede_simples()
        X, y = _dados_classificacao()
        loss, acc = rede.evaluate(X, y, cross_entropy)
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_acuracia_entre_zero_e_um(self):
        rede = _rede_simples()
        X, y = _dados_classificacao()
        _, acc = rede.evaluate(X, y, cross_entropy)
        assert 0.0 <= acc <= 1.0

    def test_acuracia_melhora_apos_treinamento(self):
        np.random.seed(1)
        rede = _rede_simples()
        X, y = _dados_classificacao(n=120)
        _, acc_antes = rede.evaluate(X, y, cross_entropy)
        rede.fit(X, y, epochs=300, taxa_aprendizado=0.05,
                 funcao_custo=cross_entropy,
                 funcao_custo_derivada=cross_entropy_derivada,
                 tamanho_lote=32, verbose=False)
        _, acc_depois = rede.evaluate(X, y, cross_entropy)
        assert acc_depois > acc_antes
