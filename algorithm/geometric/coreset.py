import numpy as np
import logging, sys
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from progress.bar import Bar

log_level = logging.DEBUG
logging.basicConfig(stream=sys.stderr, level=log_level)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
#logging.debug()
#logging.info()

# TODO
# add const c for 32

class GeometricDecomposition:
    def __init__(self, X, n, k, eps):
        self.X = X
        self.n = n
        self.k = k
        self.eps = eps

    def set_k(self, k):
        self.k = k

    def set_X(self, X):
        self.X = X

    ### Tools ###

    def _calc_nearest_neighbors(self, X, C):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(C)
        dist, indices = nbrs.kneighbors(X)
        return dist, indices
    
    def _concat(self, dist, index):
        info = np.ndarray(shape=(dist.shape), dtype=int)

        for i in range(dist.shape[0]):
            if dist[i][0] > dist[i][1]:
                info[i][0] = int(dist[i][0])
                info[i][1] = int(index[i][0])
            else:
                info[i][0] = int(dist[i][1])
                info[i][1] = int(index[i][1])
        
        return info

    def _concat_map(self, dist, index, P):
        info = np.ndarray(shape=(dist.shape[0]), dtype=int)
        point = np.ndarray(shape=(dist.shape))

        for i in range(dist.shape[0]):
            if dist[i][0] > dist[i][1]:
                info[i] = int(dist[i][0])
                point[i] = P[index[i][0]]
            else:
                info[i] = int(dist[i][1])
                point[i] = P[index[i][1]]
        
        return info, point
    
    def _calc_nearest_center(self, X, C):
        P = np.ndarray(shape=(X.shape))
        for idx, point in enumerate(X):
            dist = np.inf
            best_center = [0, 0]
            for center in C:
                new_dist = np.power(point-center, 2).sum()
                if new_dist < dist:
                    best_center = center
                    dist = new_dist
            P[idx] = best_center

        return P

    ### Implementation ###
        
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
            T = np.append(T, X[index[0]], axis=0)
            X = np.delete(X, index[0], axis=0)
            dist = np.delete(dist, index[0], axis=0)

        return T

    def _calc_L(self, T):
        X = np.ndarray.copy(self.X)
        dist = np.sqrt(np.power(X-T[-1], 2).sum(axis=1))
        return int(np.amax(dist))

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
            P = np.unique(np.concatenate((P, P_i), axis=0), axis=0)
            if (P.shape[0] >= n/2):
                break
            
        for point in P:
            index = np.where(X == point)
            if len(index[0]):
                X = np.delete(X, index[0][0], axis=0)

        return X
           
    def _handle_bad_points(self, X, C, L):
        # Calulate needed constant
        n = X.shape[0]
        beta = int(n/(10*np.log(n)))

        # Calculate nearest neighbors
        dist, index = self._calc_nearest_neighbors(X,C)
        info = self._concat(dist, index)
        R = self._find_good_set(beta, X, C, L, info)

        return R

    def _compute_fast_factor_approx_coreset(self, X):
        n = X.shape[0]

        # Computing 2-approx k-centers for k ~ (n^(1/4))
        alpha = 2
        k = int(alpha * np.ceil(np.power(n, 1/4)))
        
        A = self._farthest_point_algorithm(k)
        L = self._calc_L(A)

        # Picking random sample from X
        gamma = 3
        sample_size = int(np.ceil(gamma * k * np.log(n)))
        if sample_size >= n:
            return X
        index = np.random.choice(X.shape[0], sample_size, replace=False)
        B = X[index]

        # Approx centers set
        C = np.unique(np.concatenate((A, B), axis=0), axis=0)
        return C, L

    def _snap_point_into_fake_exp_grid(self, p, P, R):
        # Unweighted version

        ### CONSTANTS ###
        approx_factor = 32

        D = np.sqrt(np.power(P - p, 2).sum(axis=1))
        limit = int(2*np.log(approx_factor*self.n))
       
        ### plt ###
        # plt.cla()
        # if P.shape[0] > 10:
        #     plt.scatter(p[0], p[1])
        #     plt.scatter(P[:, 0], P[:, 1])
        ###########

        # init S, W 
        S = np.array([[0, 0]])
        W = np.array([0])

        # first round
        index = np.where(D < R)
        P_tmp = P[index[0]]
        D_tmp = D[index[0]]
        size = len(P_tmp)
        
        if(size > 0):
            idx = np.random.choice(size, 1, replace=False)
            S = np.append(S, P_tmp[idx], axis=0)
            W = np.append(W, [size], axis=0)

        R_tmp = R
        for i in range(limit):
            R_tmp *= 2
            index = np.where((R_tmp/2 < D) & (D < R_tmp))

            ### plt ###
            # fig = plt.Circle((p[0], p[1]), R_tmp, fill=False)
            # ax = plt.gca()
            # ax.add_artist(fig)
            ###########

            if len(index[0]) > 0:
                candidates = P[index[0]]
                size_tmp = int(np.ceil(np.log(len(index[0]))))
                idx = np.random.choice(len(candidates), size_tmp, replace=False)
                S = np.append(S, candidates[idx], axis=0)
               
        ### plt ###
        # if P.shape[0] > 10:
        #     plt.scatter(S[1:, 0], S[1:, 1])
        #     plt.show()

        return S[1:], W[1:]

    def _compute_centroid_set(self):
        # Har-Peled and Mazumdar: Coresets for k-maens nad k-median Clustering and their Application.

        ### CONSTANTS ###
        alpha = 2
        approx_factor = 32

        # Compute Fast Constant Factor Approximation Algorithm
        X = np.ndarray.copy(self.X)
        index = np.random.choice(X.shape[0], 1, replace=False)      
        A = np.array(X[index])
        X = np.delete(X, index, axis=0)

        n = X.shape[0]
        iter = np.floor(np.log(n))
        min_size = int(n/(np.log(n)))

        # Temp const for testing reasons
        k = int(alpha * np.ceil(np.power(n, 1/4)))

        while iter:
            C, L = self._compute_fast_factor_approx_coreset(X)
            A = np.unique(np.concatenate((A, C), axis=0), axis=0)
            X = self._handle_bad_points(X, C, L)
            iter -= 1
            if X.shape[0] < min_size:
                A = np.unique(np.concatenate((A, X), axis=0), axis=0)
                break

        # !!! Lost some point <1% !!!
        # Preparing for next stage of the algorithm
        X = np.ndarray.copy(self.X)
        for p in A:
            index = np.where(X == p)
            X = np.delete(X, index[0], axis=0)

        logging.info(" A size: %d", A.shape[0])
        logging.info(" Exp A size: %d", int(k*np.log(n)*np.log(n)))
        logging.info(" X size: %d", X.shape[0])
        logging.info(" A + X = %d, diff = %d", A.shape[0]+X.shape[0], self.X.shape[0] - (A.shape[0]+X.shape[0]))
        logging.info(" percentage loss = %f", (self.X.shape[0] - (A.shape[0]+X.shape[0]))*100/self.X.shape[0])

        # Compute final Coreset from Approximate Clustering 
        dist, index = self._calc_nearest_neighbors(X, A)
        _, point = self._concat_map(dist, index, A)

        value = np.power(X-point, 2).sum(axis=1).sum()
        R = int(np.sqrt(value/(approx_factor*self.n)))

        logging.info(" R value: %d", R)

        S = np.array([[0, 0]])
        W = np.array([0])

        for p in A:
            index = np.where(point == p)
            if (len(index[0]) > 0):
                P = np.unique(X[index[0]], axis=0)
                S_tmp, W_tmp = self._snap_point_into_fake_exp_grid(p, P, R)
                S = np.append(S, S_tmp, axis=0)
                W = np.append(W, W_tmp, axis=0)
                
        logging.info(" S size: %d", S.shape[0])
        return S[1:], W[1:]

    def _swap_heuristic(self, S, W, coverage):
        index = np.random.choice(S.shape[0], self.k, replace=False)
        C = S[index]
        S = np.delete(S, index, axis=0)

        # self.X <-- swap --> S
        centers = self._calc_nearest_center(self.X, C)
        cost = np.power(self.X-centers, 2).sum(axis=1).sum()

        bar = Bar('Processing', max=coverage)
        for i in range(coverage):
            index_new = np.random.choice(S.shape[0], 1, replace=False)
            index_current = np.random.choice(C.shape[0], 1, replace=False)

            point = C[index_current]
            C = np.delete(C, index_current, axis=0)
            C = np.append(C, S[index_new], axis=0)
            
            # self.X <-- swap --> S
            centers = self._calc_nearest_center(self.X, C)
            cost_new = np.power(self.X-centers, 2).sum(axis=1).sum()

            if cost_new < cost:
                S = np.delete(S, index_new, axis=0)
                S = np.append(S, point, axis=0)
                cost = cost_new
            else:
                C = np.delete(C, self.k-1, axis=0)
                C = np.append(C, point, axis=0)

            bar.next()
        bar.finish()

        return C

    def _compute_polynomial_approx(self):
        # Local search 
        # Build centroid set aka. candidate centers set C

        ### CONSTANTS ###
        c = 1.5
        log_const = 0.5
        power_const = 0.1
        coverage = 20

        d = self.X.shape[1]
        log = np.log(1/self.eps) if np.log(1/self.eps) > 0 else log_const
        power = np.power(self.eps, d) if np.power(self.eps, d) > 0 else power_const
        size = log*power*c

        # Computing centroid set
        S, W = self._compute_centroid_set()
        logging.info(" Expected centroid set: %d", size * self.X.shape[0])

        # Single/multiple-swap heuristic
        logging.info(" Single-swap heuristic - heavy computations")
        centers = self._swap_heuristic(S, W, coverage)

        return centers

    def _find_radius(self, X, center):
        radius = 0
        for point in X:
            dist = np.power(point-center, 2).sum()
            radius = dist if dist > radius else radius

        return int(np.sqrt(radius))/2

    def _compute_eps_cover(self, centers):
        C = self._calc_nearest_center(self.X, centers)
        coreset = np.array([[0, 0]])

        for center in centers:
            index = np.where(C == center)
            points = self.X[index[0]]
            D = np.sqrt(np.power(points - center, 2).sum(axis=1))

            sample_size = int(self.k*np.log(len(points)))
            radius = self._find_radius(self.X, center)
            levels = int(np.ceil(np.log(sample_size)))

            last_radius = 0
            current_radius = radius/2
            sample_size_current = sample_size*0.7
            for i in range(levels):
                index = np.where((last_radius < D) & (D < current_radius))
                sample = points[index[0]]

                sample_size_current = sample.shape[0] if sample_size_current > sample.shape[0] else sample_size_current
                index_coreset = np.random.choice(sample.shape[0], int(sample_size_current), replace=False)
                coreset = np.append(coreset, points[index_coreset], axis=0)

                last_radius = current_radius
                current_radius += radius/np.power(2,i+2)
                sample_size_current /= 2

        return coreset[1:]

    def compute(self):
        # Compute polynomial approximation a*OPT -> centers
        centers = self._compute_polynomial_approx()

        # Comupte eps ball cover for each center in centers
        coreset = self._compute_eps_cover(centers)

        return coreset