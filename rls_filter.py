import numpy as np


class RLSFilterAnalyticIntercept():
    """
    Class representing the state of a recursive least squares estimator with
    intercept estimation.
    """

    def __init__(self, input_dim, output_dim, alpha=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.param_dim = input_dim

        self.t = 0.0

        self.intercept = np.zeros((self.output_dim, 1))

        self.theta = np.zeros((self.param_dim, self.output_dim))
        self.corrected_theta = np.zeros_like(self.theta)
        self.feat_mean = np.zeros((1, self.param_dim))
        self.output_mean = np.zeros((1, self.output_dim))
        self.covar = np.eye(self.param_dim) * alpha

    def _make_feature_vec(self, in_vec):
        assert(in_vec.shape == (self.input_dim, 1))

        return in_vec.transpose()

    def _update_covar(self, U, C, V):
        assert(U.shape == (self.param_dim, 2))
        assert(C.shape == (2, 2))
        assert(V.shape == (2, self.param_dim))

        inv_part = np.linalg.inv(C) + V.dot(self.covar).dot(U)
        update = self.covar.dot(U).dot(np.linalg.inv(inv_part)).dot(V).dot(self.covar)

        self.covar = self.covar - update

    def _update_theta(self, C_t, feat, output):
        assert(feat.shape == (1, self.param_dim))
        assert(output.shape == (self.output_dim, 1))
        assert(C_t.shape == (self.param_dim, self.param_dim))

        inner_term = feat.transpose().dot(output.transpose()) - C_t.dot(self.theta)
        update = self.covar.dot(inner_term)
        self.theta = self.theta + update

    def _update_output_mean(self, output):
        assert(output.shape == (self.output_dim, 1))
        self.output_mean = self.output_mean + (1.0 / self.t) * (output.transpose() - self.output_mean)

    def _update_feat_mean(self, feat):
        assert(feat.shape == (1, self.param_dim))
        self.feat_mean = self.feat_mean + (1.0 / self.t) * (feat - self.feat_mean)

    def _make_U(self, feat):
        assert(feat.shape == (1, self.param_dim))
        return np.block([self.feat_mean.transpose(), feat.transpose()])

    def _make_V(self, feat):
        assert(feat.shape == (1, self.param_dim))
        return np.block([[self.feat_mean],[feat]])

    def _make_C(self):
        return (1 / self.t ** 2) * np.array([[((2.0 * self.t - 1.0) ** 2) - 2.0 * (self.t ** 2), -(2.0 * self.t - 1.0) * (self.t - 1.0)],
                         [-(2.0 * self.t - 1.0) * (self.t - 1.0), (self.t - 1.0) ** 2]])

    def process_datum(self, in_vec, output):
        feat = self._make_feature_vec(in_vec)
        self.t += 1.0

        if self.t == 1.0:
            self._update_feat_mean(feat)
            self._update_output_mean(output)
            return

        U = self._make_U(feat)
        V = self._make_V(feat)
        C = self._make_C()
        C_t = U.dot(C).dot(V)

        self._update_covar(U, C, V)
        self._update_output_mean(output)
        self._update_feat_mean(feat)
        self._update_theta(C_t, feat, output)
        self.corrected_theta = self.theta - ((2 * self.t - 1) * self.covar.dot(self.feat_mean.transpose()).dot(self.output_mean))
        self.intercept = (self.output_mean - self.feat_mean.dot(self.corrected_theta)).transpose()

    def get_identified_mats(self):
        """
        Returns the current estimated A, B, and c matrices.
        """
        return self.corrected_theta.transpose(), self.intercept

    def predict(self, in_vec):
        feat = self._make_feature_vec(in_vec)
        prediction = feat.dot(self.corrected_theta).transpose() + self.intercept
        assert(prediction.shape == (self.output_dim, 1))

        return prediction