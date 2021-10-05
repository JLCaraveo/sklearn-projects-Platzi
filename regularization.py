from numpy.random import RandomState
import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/raw/felicidad.csv')
    print(dataset.describe())

    #Extrayendo los features y target
    x = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(x.shape)
    print(y.shape)

    #Particionando el dataset en training y test
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    #Aplicando los regresores
    linear_model = LinearRegression()
    linear_model.fit(X_train, Y_train)
    y_predict_linear = linear_model.predict(X_test)

    #Aplicando la regularizacion Lasso
    model_lasso = Lasso(alpha=0.02)
    model_lasso.fit(X_train, Y_train)
    y_predict_lasso = model_lasso.predict(X_test)

    #Aplicando la regularizacion Ridge
    model_ridge = Ridge(alpha=1)
    model_ridge.fit(X_train, Y_train)
    y_predict_ridge = model_ridge.predict(X_test)

    #Aplicando la regularizacion ElasticNet
    elastic_net = ElasticNet(random_state=0)
    elastic_net.fit(X_train, Y_train)
    y_predict_elastic = elastic_net.predict(X_test)

    #Comparando las perdidas de los modelos aplicados
    linear_loss = mean_squared_error(Y_test, y_predict_linear)
    print('Linear Loss: {}'.format(linear_loss))

    lasso_loss = mean_squared_error(Y_test, y_predict_lasso)
    print('Lasso Loss: {}'.format(lasso_loss))

    rigde_loss = mean_squared_error(Y_test, y_predict_ridge)
    print('Ridge Loss: {}'.format(rigde_loss))

    elastic_loss = mean_squared_error(Y_test, y_predict_elastic)
    print('Elastic Loss: {}'.format(elastic_loss))

    #Imprimiento los coeficientes para cada modelo
    print('Coef Lasso: {}'.format(model_lasso.coef_))
    print('Coef Ridge: {}'.format(model_ridge.coef_))