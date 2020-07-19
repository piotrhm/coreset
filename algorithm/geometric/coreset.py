import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging, sys

log_level = logging.DEBUG
logging.basicConfig(stream=sys.stderr, level=log_level)
#logging.debug()
#logging.info()

class GeometricDecomposition:
    def __init__(self, X, k, eps):
        self.X = X
        self.k = k
        self.eps = eps

    def set_k(self, k):
        self.k = k

    def set_X(self, X):
        self.X = X

    def _farthest_point_algorithm(self, k):
        # F.Gonzalez: Custering to minimize the maximum distance
        X = np.ndarray.copy(self.X)

        index = np.random.choice(X.shape[0], 1, replace=False)      
        T = np.array(X[index])
        X = np.delete(X, index, axis=0)
        dist = np.ndarray(shape=X.shape[0])
        dist.fill(np.NINF)

        while T.size < k*2:
            dist_new = np.sqrt(np.power(X-T[-1], 2).sum(axis=1))
            dist = np.maximum(dist, dist_new)
            D = np.amax(dist)
            index = np.where(dist == D)
            T = np.append(T, X[index], axis=0)
            X = np.delete(X, index, axis=0)
            dist = np.delete(dist, index, axis=0)

        return T

    def _calc_L(self, T):
        X = np.ndarray.copy(self.X)
        dist = np.sqrt(np.power(X-T[-1], 2).sum(axis=1))
        return int(np.amax(dist))

    def _calc_nearest_neighbors(self, X, C):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(C)
        dist, indices = nbrs.kneighbors(X)
        return dist, indices

    def _find_good_set(self, beta, X, C, L, info):
        n = X.shape[0]
        index = np.random.choice(C.shape[0], 1, replace=False)      
        P = np.array(C[index])

        #P_i = P[2^(i-1)L/n, 2^(i)L/n]
        upper_bound = int(L/(4*n))

        while True:
            indicates = info[:,0] < upper_bound
            upper_bound *= 2

            P_i = X[indicates]
            P = np.concatenate((P, P_i), axis=0)

            if (P.shape[0] >= n/2):
                break
            
        for point in P:
            index = np.where(X == point)
            if len(index[0]):
                X = np.delete(X, index[0][0], axis=0) #check it !!!

        return X
           
    def _handle_bad_points(self, X, C, L):
        #calulate needed constant
        n = X.shape[0]
        beta = int(n/(10*np.log(n)))

        #calculate nearest neighbors
        dist, indices = self._calc_nearest_neighbors(X,C)
        info = np.ndarray(shape=(dist.shape), dtype=int)

        for i in range(dist.shape[0]):
            if dist[i][0] > dist[i][1]:
                info[i][0] = int(dist[i][0])
                info[i][1] = int(indices[i][0])
            else:
                info[i][0] = int(dist[i][1])
                info[i][1] = int(indices[i][1])

        R = self._find_good_set(beta, X, C, L, info)

        return R

    def _compute_fast_factor_approx_coreset(self, X):
        n = X.shape[0]

        # computing 2-approx k-centers for k ~ (n^(1/4))
        alpha = 2
        k = int(alpha * np.ceil(np.power(n, 1/4)))
        
        A = self._farthest_point_algorithm(k)
        L = self._calc_L(A)

        # picking random sample from X
        gamma = 2
        sample_size = int(np.ceil(gamma * k * np.log(n)))
        if sample_size >= n:
            return X
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        B = X[indices]

        # approx centers set
        C = np.concatenate((A, B), axis=0)
        return C, L

    def _compute_approx_clutering_coreset(self):
        return 0

    def _compute_centroid_set(self):
        # Har-Peled and Mazumdar: Coresets for k-maens nad k-median Clustering and their Application.

        # Compute Fast Constant Factor Approximation Algorithm
        X = np.ndarray.copy(self.X)
        index = np.random.choice(X.shape[0], 1, replace=False)      
        A = np.array(X[index])
        X = np.delete(X, index, axis=0)

        n = X.shape[0]
        iter = np.floor(np.log(n))
        min_size = int(n/(np.log(n)))

        while iter:
            C, L = self._compute_fast_factor_approx_coreset(X)
            A = np.concatenate((A, C), axis=0)
            X = self._handle_bad_points(X, C, L)
            iter -= 1
            if X.shape[0] < min_size:
                A = np.concatenate((A, X), axis=0)
                break

        logging.info(" A size: %d", A.shape[0])
        logging.info(" Exp A size: %d", int(15*np.log(n)*np.log(n)))

        # Compute final Coreset from Approximate Clustering 

        # TBA

        return A

    def _swap_heuristic(self):
        return 0

    def _compute_polynomial_approx(self):
        # Local search 
        # - Build centroid set aka. candidate centers set C == computing corest with large k
        # - Single/multiple-swap heuristic 
        return 0

    def compute(self):
        # Compute polynomial approximation a*OPT -> C
        # Comupte eps ball cover for each center in C
        return 0