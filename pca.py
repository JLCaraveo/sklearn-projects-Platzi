import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df_heart = pd.read_csv('./data/raw/heart.csv')
    print(df_heart.head()) 

    #Obteniendo los features
    df_features = df_heart.drop(['target'], axis=1)
    #Obteniendo el target
    df_target = df_heart['target']

    #Normalizando los datos (transformandolos para su procesamiento)
    df_features = StandardScaler().fit_transform(df_features)

    #Obteniendo el conjunto de entrenamiento y el de testing
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    #Comparando el conjunto de datos
    print(X_train.shape)
    print(Y_train.shape)

    #Configurando el algoritmo PCA
    #n_components por default = min(n_muestras, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    #Configurando el algoritmo IPCA
    #n_components por default = min(n_muestras, n_features)
    #IPCA no manda a entrenar todos los datos al mismo tiempo, sino por bloques, es decir, por batch poco a poco
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    #Configurando la regresion logistica con PCA
    logistic = LogisticRegression(solver='lbgfgs')

    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)

    logistic.fit(df_train, Y_train)

    #Calculando la efectividad del modelo con PCA
    print('Score PCA: {}'.format(logistic.score(df_test, Y_test)))

    #Configurando la regresion logistica con IPCA
    logistic_ipca = LogisticRegression(solver='lbgfgs')

    df_train_ipca = ipca.transform(X_train)
    df_test_ipca = ipca.transform(X_test)

    logistic_ipca.fit(df_train, Y_train)

    #Calculando la efectividad del modelo con IPCA
    print('Score IPCA: {}'.format(logistic_ipca.score(df_test, Y_test)))
