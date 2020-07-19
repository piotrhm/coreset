# coreset
Collection of algorithms for coreset problem. Work done in this repository is part of a my bachelor thesis. Project under the supervision of the faculty member from [Jagiellonian University](https://uj.edu.pl), [Theoretical Computer Science Department](https://tcs.uj.edu.pl).

Piotr Helm

## Algorithms

### Geometric Decomposition

Implemented algorithms:

Farthest Point Algorithm from [3][4]
Fast Constant Factor Approximation from [2]

For computing nearest neighbors I used sklearn classifier.
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

### Lightweight Coreset

Implemented algorithms:

Lightweight Coreset from [1]

## Example

Simple testing can be found under example*.py files. 

For a reference model I used Kmeans implementation from sklearn as it provides best performance and scalability.
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html

## References

[1] Olivier Bachem, Mario Lucic, and Andreas Krause. 2018. Scalable k -Means Clustering via Lightweight Coresets. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18). Association for Computing Machinery, New York, NY, USA, 1119–1127. DOI:https://doi.org/10.1145/3219819.3219973

[2] Har-Peled, S., & Mazumdar, S. (2004). On coresets for k-means and k-median clustering. Conference Proceedings of the Annual ACM Symposium on Theory of Computing, 291-300. https://doi.org/10.1145/1007352.1007400

[3] T. Feder and D. H. Greene. Optimal algorithms for approximate clustering. In Proc. 20th Annu. ACM Sympos. Theory Comput., pages 434–444, 1988.

[4]  T. Gonzalez. Clustering to minimize the maximum intercluster distance. Theoret. Comput. Sci., 38:293–306, 1985.