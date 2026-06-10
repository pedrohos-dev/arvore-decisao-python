from experimentos import rodar_parte_1, rodar_parte_2, rodar_parte_3, rodar_parte_4

def main():
    print("=" * 60)
    print("INICIANDO EXECUÇÃO DA PRÁTICA DE LABORATÓRIO")
    print("=" * 60)
    
    # Executa a Parte 1 (Árvore de Decisão Padrão)
    rodar_parte_1()
    
    # Executa a Parte 2 (Random Forest Padrão)
    rodar_parte_2()
    
    # Executa a Parte 3 (Variação de Profundidade) e captura o melhor parâmetro
    melhor_profundidade = rodar_parte_3()
    
    # Executa a Parte 4 (Variação do N de estimadores) usando a melhor prof. encontrada
    rodar_parte_4(melhor_profundidade)
    
    print("\n" + "=" * 60)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("Tudo foi impresso acima. Os GIFs animados foram salvos na pasta.")
    print("=" * 60)

if __name__ == "__main__":
    main()