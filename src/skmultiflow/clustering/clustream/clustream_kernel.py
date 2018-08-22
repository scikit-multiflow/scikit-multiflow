from skmultiflow.clustering.cluster_feature import CFCluster
import numpy as np

EPSILON = 0.00005
MIN_VARIANCE = 1e-50


class ClustreamKernel(CFCluster):

    def __init__(self, X=None, weight=None, cluster=None, dimensions=None, timestamp=None, T=None, M=None):
        self.T = T
        self.M = M
        if X is not None and weight is not None and dimensions is not None:
            super().__init__(X=X, weight=weight, dimensions=dimensions)
            self.LST = timestamp * weight
            self.SST = timestamp * timestamp * weight
        elif cluster is not None:
            super().__init__(cluster=cluster)
            self.LST = cluster.LST
            self.SST = cluster.SST

    def get_center(self):
        assert not self.is_empty()
        res = [0.0] * len(self.LS)
        for i in range(len(res)):
            res[i] = self.LS[i] / self.N
        return res

    def is_empty(self):
        return self.N == 0

    def get_radius(self):
        if self.N == 1:
            return 0
        return self.get_deviation() * self.T

    def get_deviation(self):
        variance = self.get_variance_vector()
        sum_of_deviation = 0.0
        for i in range(len(variance)):
            d = np.sqrt(variance[i])
            sum_of_deviation += d
        return sum_of_deviation / len(variance)

    def get_variance_vector(self):
        res = [0.0] * len(self.LS)
        for i in range(len(self.LS)):
            ls = self.LS[i]
            ss = self.SS[i]
            ls_div_n = ls / self.get_weight()
            ls_div_n_squared = ls_div_n * ls_div_n
            ss_div_n = ss / self.get_weight()
            res[i] = ss_div_n - ls_div_n_squared

            if res[i] <= 0.0:
                if res[i] > - EPSILON:
                    res[i] = MIN_VARIANCE
        return res

    def insert(self, X, weight, timestamp):
        self.N += weight
        self.LST += timestamp * weight
        self.SST += timestamp * weight
        for i in range(len(X)):
            self.LS[i] += X[i] * weight
            self.SS[i] += X[i] * X[i] * weight

    def get_relevance_stamp(self):
        if self.N < 2 * self.M:
            return self.get_mu_time()
        return self.get_mu_time() + self.get_sigma_time() * self.get_quantile(float(self.M)/(2 * self.N))

    def get_mu_time(self):
        return self.LST/self.N

    def get_sigma_time(self):
        return np.sqrt(self.SST/self.N - (self.LST/self.N) * (self.LST/self.N))

    def get_quantile(self, z):
        assert (z >= 0 and z <= 1)
        return np.sqrt(2) * self.inverse_error(2 * z - 1)

    @staticmethod
    def inverse_error(x):
        z = np.sqrt(np.pi) * x
        res = z / 2
        z2 = z * z

        zprod = z2 * z
        res += (1.0/24) * zprod

        zprod *= z2 # z5
        res += (7.0/960) * zprod

        zprod *= z2 # z ^ 7
        res += (127 * zprod) / 80640

        zprod *= z2 # z ^ 9
        res += (4369 * zprod) / 11612160

        zprod *= z2 # z ^ 11
        res += (34807 * zprod) / 364953600

        zprod *= z2 # z ^ 13
        res += (20036983 * zprod) / 797058662400

        return res

    def add(self, cluster):
        assert len(cluster.LS) == len(self.LS)
        self.N += cluster.N
        self.LST += cluster.LST
        self.SST += cluster.SST
        self.add_vectors(self.LS, cluster.LS)
        self.add_vectors(self.SS, cluster.SS)

    def get_CF(self):
        return self

    def sample(self, random_state):
        pass