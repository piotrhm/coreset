import common.input as input
import algorithm.lightweight.coreset as alc
import common.utils as utils
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans

data = input.parse_txt("dataset/s-set/s2.txt")
lwcs = alc.LightweightCoreset(data, 1, 15, 0.001)
coreset, weights = lwcs.calc()

@utils.timeit
def test_no_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(data)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])

@utils.timeit
def test_coreset():
    kmeans = KMeans(n_clusters=15, random_state = 0).fit(coreset)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])


test_no_coreset()
test_coreset()

plt.show()