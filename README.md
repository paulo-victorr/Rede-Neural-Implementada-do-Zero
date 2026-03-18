# Rede Neural do Zero

Implementação de uma rede neural densa (MLP) **do zero**, sem frameworks de machine learning. Todo o mecanismo de forward pass, backpropagation e gradiente descendente foi escrito manualmente usando apenas NumPy.

Projeto de portfólio acadêmico — Ciência de Dados / IA, UFPB.

---

## Motivação

A maioria dos projetos de redes neurais usa PyTorch ou TensorFlow, que abstraem completamente o funcionamento interno do modelo. O objetivo aqui é o oposto: implementar cada peça à mão para demonstrar entendimento profundo de:

- **Forward pass** — como os dados fluem pela rede camada a camada
- **Backpropagation** — como o gradiente do erro é propagado de volta
- **Gradiente descendente com mini-batches** — como os pesos são atualizados
- **Funções de ativação e de custo** — e por que cada derivada tem a forma que tem

---

## Arquitetura da rede (MNIST)

```
Entrada: 784 features (pixels 28×28 normalizados)
    ↓
Camada 1: 128 neurônios · ReLU
    ↓
Camada 2: 64 neurônios · ReLU
    ↓
Saída: 10 neurônios · Softmax (dígitos 0–9)
```

| Hiperparâmetro    | Valor  |
|-------------------|--------|
| Épocas            | 20     |
| Taxa de aprendizado | 0.01 |
| Tamanho do lote   | 64     |
| Inicialização     | He     |
| Função de custo   | Cross-entropy categórica |

---

## Resultados no MNIST

| Conjunto    | Loss   | Acurácia |
|-------------|--------|----------|
| Validação   | —      | ~97%     |
| Teste       | —      | ~97%     |

> Os valores exatos são gerados ao executar o notebook `notebooks/mnist_demo.ipynb`.

---

## Estrutura do projeto

```
Rede_Neural_do_Zero/
├── src/
│   ├── activations.py   # ReLU, Sigmoid, Softmax e derivadas
│   ├── losses.py        # MSE, Cross-entropy e derivadas
│   ├── layers.py        # Classe Layer: forward + backward
│   ├── network.py       # Classe Network: fit, predict, evaluate
│   └── utils.py         # one_hot, normalizar, dividir
├── tests/
│   ├── test_activations.py
│   ├── test_losses.py
│   ├── test_layers.py
│   ├── test_network.py
│   └── test_utils.py
├── notebooks/
│   └── mnist_demo.ipynb  # Demo completa com MNIST
├── data/                 # Dataset baixado automaticamente pelo notebook
├── CLAUDE.md
└── README.md
```

---

## Tecnologias

| Biblioteca  | Uso |
|-------------|-----|
| Python 3.11+ | Linguagem principal |
| NumPy | Operações matriciais (único uso permitido de lib externa para cálculos) |
| Matplotlib | Gráficos e visualizações no notebook |
| pytest | Testes unitários |

**Não utiliza:** scikit-learn · PyTorch · TensorFlow · Keras

---

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/rede-neural-do-zero.git
cd rede-neural-do-zero

# (Opcional) Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instale as dependências
pip install numpy matplotlib pytest
```

---

## Como rodar

### Testes unitários

```bash
pytest tests/ -v
```

Saída esperada: **68 testes passando** cobrindo ativações, losses, camadas, rede e utilitários.

### Demo com MNIST

Abra e execute todas as células do notebook:

```bash
jupyter notebook notebooks/mnist_demo.ipynb
```

O notebook baixa o MNIST automaticamente na primeira execução (sem sklearn — leitura manual do formato IDX binário).

### Verificação de tipos

```bash
mypy src/
```

---

## Módulos principais

### `src/activations.py`
Funções de ativação e suas derivadas: `relu`, `sigmoid`, `softmax`.
A derivada da softmax não é implementada separadamente — ela se cancela com a cross-entropy no backward, resultando no gradiente simplificado `ŷ − y`.

### `src/losses.py`
Funções de custo: `mse` e `cross_entropy`.
`cross_entropy_derivada` retorna o gradiente combinado Softmax + CE: `(ŷ − y) / n`.

### `src/layers.py`
Classe `Layer` com:
- `forward(entrada)` — computa `z = XW + b` e aplica ativação; armazena `_entrada` e `_z` para o backward
- `backward(grad_saida)` — calcula `grad_W`, `grad_b` e propaga o gradiente para a camada anterior
- `atualizar_pesos(lr)` — gradiente descendente: `W -= lr * grad_W`

### `src/network.py`
Classe `Network` que orquestra as camadas:
- `fit(X, y, ...)` — loop de épocas com mini-batches e embaralhamento
- `predict(X)` — forward pass puro
- `evaluate(X, y, ...)` — retorna loss e acurácia

### `src/utils.py`
- `one_hot(y, n_classes)` — codificação one-hot
- `normalizar(X)` — Z-score por feature
- `dividir(X, y, proporcao_treino)` — split treino/validação com semente
