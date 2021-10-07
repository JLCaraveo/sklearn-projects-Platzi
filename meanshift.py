import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    df_candies = pd.read_csv('./data/raw/candy.csv')

    x = df_candies.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(x)
    print(meanshift.labels_)
    print('_'*64)
    print(meanshift.cluster_centers_)