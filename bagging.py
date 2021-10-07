import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df_heart = pd.read_csv('./data/raw/heart.csv')

    x = df_heart.drop(['target'], axis=1)
    y = df_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35)

    knn_class = KNeighborsClassifier().fit(x_train, y_train)

    knn_pred = knn_class.predict(x_test)

    print('KNN Accuracy: %f'%accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(x_train, y_train)

    bag_pred = bag_class.predict(x_test)

    print('BAG Accuracy: %f'%accuracy_score(bag_pred, y_test))