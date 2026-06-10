import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def carregar_dados():
    """Carrega a base de dados de diabetes do scikit-learn."""
    diabetes = load_diabetes()
    return diabetes.data, diabetes.target

def executar_experimento(modelo_classe, kwargs_modelo, n_divisoes=10, test_size=0.2):
    """Executa o modelo por N divisões diferentes de treino/teste."""
    X, y = carregar_dados()
    maes_treino = []
    maes_teste = []
    
    ultimo_modelo = None
    X_treino, X_teste, y_treino, y_teste = (None, None, None, None)

    for i in range(n_divisoes):
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y, test_size=test_size, random_state=i
        )
        
        modelo = modelo_classe(**kwargs_modelo, random_state=i)
        modelo.fit(X_treino, y_treino)
        
        maes_treino.append(mean_absolute_error(y_treino, modelo.predict(X_treino)))
        maes_teste.append(mean_absolute_error(y_teste, modelo.predict(X_teste)))
        
        ultimo_modelo = modelo

    return maes_treino, maes_teste, ultimo_modelo, X_teste, y_teste