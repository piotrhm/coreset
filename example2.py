import common.input as input
import algorithm.geometric.coreset as agc
import common.utils as utils
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

data = input.parse_txt("dataset/s-set/s3.txt")
opt = input.parse_txt("dataset/s-set/s3-label.pa")
centers = input.parse_txt("dataset/s-set/s3-cb.txt")

geo = agc.GeometricDecomposition(data, 15, 0.1)
geo._compute_centroid_set()
