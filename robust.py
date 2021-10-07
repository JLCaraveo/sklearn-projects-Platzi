import pandas as pd
from sklearn.linear_model import RANSACRegressor, HuberRegressor

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/raw/felicidad.csv')
    print(dataset.head())

    x = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

    estimators = {
        'SVR' : SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC' : RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        print('_'*64)
        print(name)
        print('MSE: %f'%mean_squared_error(y_test, predictions))