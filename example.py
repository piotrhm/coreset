import common.input as input
import algorithm.lightweight.coreset as alc
import common.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import statistics

from sklearn.cluster import KMeans

data = input.parse_txt("dataset/s-set/s3.txt")
opt = input.parse_txt("dataset/s-set/s3-label.pa")
centers = input.parse_txt("dataset/s-set/s3-cb.txt")

#Computing lightweight coreset
lwcs = alc.LightweightCoreset(data, 15, 0.1)
coreset, weights = lwcs.compute()


@utils.timeit
def test_no_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(X=data)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    cost = utils.cost_function(data, kmeans.labels_, kmeans.cluster_centers_)
    return cost

@utils.timeit
def test_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(X=coreset, sample_weight=weights)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    cost = utils.cost_function(data, kmeans.predict(X=data), kmeans.cluster_centers_)
    return cost


cost = test_no_coreset()

results = []
for i in range(100):
    coreset, weights = lwcs.compute()
    cost_cs = test_coreset()
    results.append((cost-cost_cs)/cost)


print(statistics.mean(results))


# print("cost no coreset ", cost)
# print("cost coreset ", cost_cs)
# print("coreset improvment: {:.1%} ".format(np.abs(cost-cost_cs)/cost))