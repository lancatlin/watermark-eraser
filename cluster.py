import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from utils import parse_array
from plot import plot_3d


def to_ndarray(series):
    return np.array(series.tolist())


class ColorCluster:
    def __init__(self, df):
        self.df = df
        self.clustering = None
        self.labels = None

    def cluster(self, key, weight_key="count"):
        colors = to_ndarray(self.df[key])
        dbscan = DBSCAN(eps=5, min_samples=5)
        self.clustering = dbscan.fit(colors, sample_weight=self.df[weight_key])
        self.labels = self.clustering.labels_
        return self.labels

    def mean(self, key):
        colors = to_ndarray(self.df[key])
        label = self.clustering.labels_
        mean = []
        variance = []
        for i in range(self.clustering.labels_.max() + 1):
            mean.append(np.mean(colors[label == i], axis=0))
            variance.append(np.var(colors[label == i], axis=0))
        return mean, variance


if __name__ == "__main__":
    filename = sys.argv[1]
    df = pd.read_csv(
        filename, converters={"mixed": parse_array, "background": parse_array}
    )
    cluster = ColorCluster(df)
    result = cluster.cluster("background")
    print(result)
    plot_3d(df, cluster.clustering.labels_)
    print(cluster.mean("mixed"))
    print(cluster.mean("background"))
    plt.show()
