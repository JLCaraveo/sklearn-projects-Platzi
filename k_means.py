import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

if __name__ == '__main__':
    df_candies = pd.read_csv('./data/raw/candy.csv')

    x = df_candies.drop('competitorname', axis=1)

    mini_kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x)

    mini_kmeans_pred = mini_kmeans.predict(x)

    print(mini_kmeans_pred)

    kmeans = KMeans(n_clusters=4).fit(x)
    kmeans_pred = kmeans.predict(x)
    print(kmeans_pred)

    compare = [1 if mini_kmeans_pred[i] == kmeans_pred[i] else 0 for i in range(0,len(x))]

    print(compare)