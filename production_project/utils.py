import pandas as pd
import joblib


class Utils:

    def load_from_csv(self, path):
        return pd.read_csv(path)


    def load_from_mysql(self):
        pass


    def features_target(self, df, drop_columns, target):
        x = df.drop(drop_columns, axis=1)
        y = df[target]

        return x, y


    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')
