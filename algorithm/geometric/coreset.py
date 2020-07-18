import numpy as np

class GeometricDecomposition:
    def __init__(self, X, k, eps):
        #check dtype!!!
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
           
    def _handle_bad_points(self, C, L):
        X = np.ndarray.copy(self.X)
        n = X.shape[0]
        return C

    def _compute_fast_factor_approx_coreset(self):
        X = self.X
        n = X.shape[0]

        # computing 2-approx k-centers for k ~ (n^(1/4))
        alpha = 10
        k = alpha * np.ceil(np.power(n, 1/4))
        A = self._farthest_point_algorithm(k)
        L = self._calc_L(A)
        print(L)

        # picking random sample from X
        gamma = 2
        sample_size = int(np.ceil(gamma * k * np.log(n)))
        if sample_size >= n:
            return X
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        B = X[indices]

        # approx centers set
        C = np.concatenate((A, B), axis=0)

        X_final = self._handle_bad_points(C, L)
        return X_final

    def _compute_approx_clutering_coreset(self):

        return 0

    def _compute_centroid_set(self):
        # Har-Peled and Mazumdar: Coresets for k-maens nad k-median Clustering and their Application.
        # Compute Fast Constant Factor Approximation Algorithm
        # Compute final Coreset from Approximate Clustering 
        A = self._compute_fast_factor_approx_coreset()

        return 0

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