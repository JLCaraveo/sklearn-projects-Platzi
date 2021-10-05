import pandas as pd
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

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

    #Aplicando el algoritmo kpca
    #El numero de componentes es opcional
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)
    #Tranformando los datos
    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)
    #Regresion logistica
    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(df_train, Y_train)

    #Comprobando la efectividad
    print('Score kpca: {}'.format(logistic.score(df_test, Y_test)))
