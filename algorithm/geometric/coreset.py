class GeometricDecomposition:
    def __init__(self, X, k, eps):
        self.X = X
        self.k = k
        self.eps = eps

    def set_k(self, k):
        self.k = k

    def set_X(self, X):
        self.X = X

    def compute_10_opt_approx(self):
        return 0