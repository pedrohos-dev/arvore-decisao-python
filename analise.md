# Prática de Laboratório: Algoritmos de Regressão (Diabetes Dataset)

Este projeto aplica modelos de Aprendizado de Máquina (**Árvores de Decisão** e **Florestas Aleatórias**) para prever a progressão da diabetes com base em dados clínicos de pacientes. 

O objetivo principal é avaliar o erro dos modelos utilizando a métrica **MAE (Mean Absolute Error - Erro Médio Absoluto)** e entender conceitos como **Overfitting (Sobreajuste)** e a influência dos hiperparâmetros.

---

## 📊 Entendendo os Resultados Obtidos

### Parte 1: Árvore de Decisão Simples
* **Média do MAE de Teste:** 64.35
* **MAE de Treino:** 0.00 (em todas as divisões)

**O que isso significa?** A árvore de decisão sem limite de profundidade decorou perfeitamente os dados de treino (erro zero). Porém, quando foi testada com dados novos, o erro subiu para 64.35. Isso é um caso clássico de **Overfitting** (o modelo ficou complexo demais e perdeu a capacidade de generalizar).

### Parte 2: Floresta Aleatória (Random Forest)
* **Média do MAE de Teste:** 45.91
* **Desvio Padrão:** 2.68

**O que isso significa?**
A Floresta Aleatória combina várias árvores de decisão. Ao fazer isso, o erro no teste caiu drasticamente de **64.35** para **45.91**. O modelo se tornou muito mais robusto e estável (baixo desvio padrão).

### Parte 3: O Efeito da Profundidade da Árvore
Aqui limitamos a altura máxima da árvore (`max_depth`) de 1 até 10 para ver onde ela performava melhor:
* Com profundidade **1**, a árvore é simples demais.
* Com profundidade **2**, encontramos o ponto ideal: o menor erro de teste (**48.54**).
* A partir da profundidade **3**, o erro de treino cai, mas o erro de teste começa a subir. A árvore começa a sofrer Overfitting novamente.

> **Resultado:** A melhor profundidade automática identificada foi **2**.

### Parte 4: Quantidade de Árvores na Floresta
Fixando a profundidade em **2** (o melhor resultado anterior), variamos o número de árvores na floresta:
* **5 árvores:** MAE 47.20
* **40 árvores:** MAE 46.06 (Melhor resultado)
* **80 árvores:** MAE 46.12

**O que isso significa?**
Aumentar a quantidade de árvores ajuda a estabilizar e reduzir o erro do modelo, mas chega a um ponto (por volta de 40 árvores) onde o ganho de performance estabiliza e colocar mais árvores não traz vantagens significativas.

---

## 📁 Estrutura do Projeto

* `utils.py`: Contém as funções de infraestrutura (carregamento da base de dados e a função que divide os dados em 80% treino e 20% teste 10 vezes mudando o `random_state`).
* `experimentos.py`: Lógica das 4 partes da prática, tabelas do terminal e renderização dos gráficos dinâmicos.
* `main.py`: O executor do projeto.
* `evolucao_profundidade.gif`: Gráfico animado mostrando as curvas de erro de Treino vs. Teste mudando a profundidade.
* `evolucao_floresta.gif`: Gráfico animado mostrando o comportamento do erro de teste conforme adicionamos mais árvores.

---

## 🛠️ Como rodar e visualizar

1. Instale os requisitos necessários:
   ```bash
   pip install scikit-learn matplotlib numpy pillow