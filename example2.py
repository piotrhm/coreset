import common.input as input
import algorithm.geometric.coreset as agc
import common.utils as utils
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

data = input.parse_txt("dataset/s-set/s3.txt")
opt = input.parse_txt("dataset/s-set/s3-label.pa")
centers = input.parse_txt("dataset/s-set/s3-cb.txt")

#Computing geometric decomposition coreset
geo = agc.GeometricDecomposition(data, 5000, 15, 0.5)
coreset = geo.compute()

@utils.timeit
def test_no_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(X=data)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    cost = utils.cost_function(data, kmeans.labels_, kmeans.cluster_centers_)
    return cost

@utils.timeit
def test_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(X=coreset)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
    cost = utils.cost_function(data, kmeans.predict(X=data), kmeans.cluster_centers_)
    return cost


cost = test_no_coreset()
cost_cs = test_coreset()
cost_opt = utils.cost_function(data, opt, centers)

print("cost no coreset ", cost)
print("cost coreset ", cost_cs)
print("coreset improvment: {:.1%} ".format(np.abs(cost-cost_cs)/cost))

#plt.show()