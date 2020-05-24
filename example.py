import common.input as input
import algorithm.lightweight.coreset as alc
import common.utils as utils
import matplotlib.pyplot as plt

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
cost_cs = test_coreset()
cost_opt = utils.cost_function(data, opt, centers)

print(cost, cost_cs, cost_opt)
print(cost_cs/cost)
print(cost-cost_cs)
print("no coreset: {:.1%} ".format(cost/cost_opt))
print("coreset: {:.1%} ".format(cost_cs/cost_opt))

#plt.show()