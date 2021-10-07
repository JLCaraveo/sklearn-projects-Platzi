import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df_heart = pd.read_csv('./data/raw/heart.csv')

    x = df_heart.drop(['target'], axis=1)
    y = df_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35)

    boost_class = GradientBoostingClassifier( n_estimators=50).fit(x_train, y_train)

    boost_pred = boost_class.predict(x_test)

    print('BAG Accuracy: %f'%accuracy_score(boost_pred, y_test))