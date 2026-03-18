# Projeto: Rede Neural do Zero
Implementação de uma rede neural densa (MLP) do zero em Python puro,
sem bibliotecas de ML externas. Portfólio acadêmico — DS/AI, UFPB.

## Objetivo
Demonstrar entendimento profundo de backpropagation, gradiente descendente
e forward pass implementando tudo manualmente com NumPy apenas.

## Stack permitida
- Python 3.11+
- NumPy (apenas para operações matriciais)
- Matplotlib (apenas para gráficos finais)
- pytest (apenas para testes)
- PROIBIDO: scikit-learn, PyTorch, TensorFlow, Keras

## Estrutura do projeto
```
neural-network-from-scratch/
├── CLAUDE.md
├── .claudeignore
├── src/
│   ├── activations.py     # ReLU, Sigmoid, Softmax e derivadas
│   ├── layers.py          # Classe Layer (forward + backward)
│   ├── network.py         # Classe Network (fit, predict, evaluate)
│   ├── losses.py          # MSE, CrossEntropy e derivadas
│   └── utils.py           # one_hot, normalize, split
├── tests/
│   ├── test_activations.py
│   ├── test_layers.py
│   └── test_network.py
├── notebooks/
│   └── mnist_demo.ipynb   # Demo final com MNIST
├── data/                  # datasets (ignorar no contexto)
└── README.md
```

## Ordem de implementação (SIGA ESSA ORDEM)
1. activations.py — funções e derivadas
2. losses.py — funções de custo e derivadas
3. layers.py — forward e backward de uma camada
4. network.py — orquestra camadas, fit e predict
5. testes unitários para cada módulo
6. notebook de demonstração com MNIST

## Estilo de código
- Português nos comentários e docstrings
- Docstrings curtas (1-2 linhas)
- Type hints em todas as funções
- Funções pequenas e com responsabilidade única
- Sem over-engineering — é portfólio, não produção

## Comandos úteis
- Rodar testes: `pytest tests/ -v`
- Verificar tipos: `mypy src/`

## RESTRIÇÕES IMPORTANTES
- Não ler arquivos em data/
- Não criar abstrações desnecessárias
- Não adicionar dependências sem perguntar
- Não gerar código para todos os módulos de uma vez
- Implementar um módulo por vez, testar, depois avançar