import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import executar_experimento

def rodar_parte_1():
    print("\n" + "="*50)
    print("PARTE 1: Árvore de Decisão")
    print("="*50)
    
    maes_treino, maes_teste, ultimo_mod, X_test, y_test = executar_experimento(
        DecisionTreeRegressor, kwargs_modelo={}
    )
    
    print(f"{'Divisão':<10} | {'MAE Treino':<12} | {'MAE Teste':<12}")
    print("-" * 40)
    for idx, (treino, teste) in enumerate(zip(maes_treino, maes_teste), 1):
        print(f"Divisão {idx:02d} | {treino:<12.2f} | {teste:<12.2f}")
        
    print("-" * 40)
    print(f"Média MAE (Teste): {np.mean(maes_teste):.2f}")
    print(f"Desvio Padrão MAE (Teste): {np.std(maes_teste):.2f}")
    
    print("\n5 Exemplos de dados reais vs previstos (Último Modelo):")
    print("-" * 40)
    predicoes = ultimo_mod.predict(X_test[:5])
    for i, (real, prev) in enumerate(zip(y_test[:5], predicoes), 1):
        print(f"Exemplo {i}: Real = {real:6.1f} | Previsto = {prev:6.1f}")

def rodar_parte_2():
    print("\n" + "="*50)
    print("PARTE 2: Floresta Aleatória (Random Forest)")
    print("="*50)
    
    maes_treino, maes_teste, _, _, _ = executar_experimento(
        RandomForestRegressor, kwargs_modelo={}
    )
    
    print(f"{'Divisão':<10} | {'MAE Treino':<12} | {'MAE Teste':<12}")
    print("-" * 40)
    for idx, (treino, teste) in enumerate(zip(maes_treino, maes_teste), 1):
        print(f"Divisão {idx:02d} | {treino:<12.2f} | {teste:<12.2f}")
        
    print("-" * 40)
    print(f"Média MAE (Teste): {np.mean(maes_teste):.2f}")
    print(f"Desvio Padrão MAE (Teste): {np.std(maes_teste):.2f}")

def rodar_parte_3():
    print("\n" + "="*50)
    print("PARTE 3: Efeito da Profundidade (Árvore de Decisão)")
    print("="*50)
    
    profundidades = list(range(1, 11))
    medias_treino, desvios_treino = [], []
    medias_teste, desvios_teste = [], []
    
    print(f"{'Prof.':<6} | {'Média Treino':<13} | {'Média Teste':<13} | {'DP Teste':<10}")
    print("-" * 52)
    
    for prof in profundidades:
        maes_tr, maes_te, _, _, _ = executar_experimento(
            DecisionTreeRegressor, kwargs_modelo={'max_depth': prof}
        )
        
        m_tr, m_te, d_te = np.mean(maes_tr), np.mean(maes_te), np.std(maes_te)
        medias_treino.append(m_tr)
        desvios_treino.append(np.std(maes_tr))
        medias_teste.append(m_te)
        desvios_teste.append(d_te)
        
        print(f"{prof:<6} | {m_tr:<13.2f} | {m_te:<13.2f} | {d_te:<10.2f}")
        
    melhor_idx = np.argmin(medias_teste)
    melhor_prof = profundidades[melhor_idx]
    print("-" * 52)
    print(f"Melhor profundidade identificada: {melhor_prof} (Menor MAE de Teste: {medias_teste[melhor_idx]:.2f})")
    
    # Criando e salvando a animação da Parte 3
    print("\n-> Gerando arquivo 'evolucao_profundidade.gif'...")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 75)
    ax.set_xlabel('Profundidade da Árvore')
    ax.set_ylabel('Erro Médio Absoluto (MAE)')
    ax.set_title('Efeito da Profundidade da Árvore no MAE')
    ax.grid(True)
    
    line_tr, = ax.plot([], [], '-o', color='blue', label='Treino')
    line_te, = ax.plot([], [], '-s', color='orange', label='Teste')
    ax.legend()

    def update(frame):
        line_tr.set_data(profundidades[:frame+1], medias_treino[:frame+1])
        line_te.set_data(profundidades[:frame+1], medias_teste[:frame+1])
        return line_tr, line_te

    ani = FuncAnimation(fig, update, frames=len(profundidades), blit=True, repeat=False)
    ani.save('evolucao_profundidade.gif', writer='pillow', fps=2)
    plt.close()
    
    return melhor_prof

def rodar_parte_4(melhor_prof):
    print("\n" + "="*50)
    print(f"PARTE 4: Número de Árvores (Profundidade Fixa = {melhor_prof})")
    print("="*50)
    
    n_estimadores = [5, 10, 20, 40, 80]
    medias_teste, desvios_teste = [], []
    
    print(f"{'Árvores':<8} | {'Média MAE Teste':<17} | {'Desvio Padrão':<13}")
    print("-" * 45)
    
    for n_est in n_estimadores:
        _, maes_te, _, _, _ = executar_experimento(
            RandomForestRegressor, kwargs_modelo={'max_depth': melhor_prof, 'n_estimators': n_est}
        )
        m_te, d_te = np.mean(maes_te), np.std(maes_te)
        medias_teste.append(m_te)
        desvios_teste.append(d_te)
        
        print(f"{n_est:<8} | {m_te:<17.2f} | {d_te:<13.2f}")
        
    print("-" * 45)
    
    # Criando e salvando a animação da Parte 4
    print("-> Gerando arquivo 'evolucao_floresta.gif'...")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 90)
    ax.set_ylim(40, 60)
    ax.set_xlabel('Número de Árvores na Floresta')
    ax.set_ylabel('Erro Médio Absoluto (MAE)')
    ax.set_title(f'Efeito da Quantidade de Árvores (Profundidade={melhor_prof})')
    ax.grid(True)
    
    line_te, = ax.plot([], [], '-o', color='green', label='Teste')
    ax.legend()

    def update(frame):
        line_te.set_data(n_estimadores[:frame+1], medias_teste[:frame+1])
        return (line_te,)

    ani = FuncAnimation(fig, update, frames=len(n_estimadores), blit=True, repeat=False)
    ani.save('evolucao_floresta.gif', writer='pillow', fps=1.5)
    plt.close()