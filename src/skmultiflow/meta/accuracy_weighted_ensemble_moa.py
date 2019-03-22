from skmultiflow.core.base import StreamModel
from skmultiflow.utils import check_random_state
from sklearn.model_selection import KFold
import copy as cp
import numpy as np

class AccuracyWeightedEnsembleMOA(StreamModel):
    """
    A direct translation of AWE from MOA to compare the performance
    """

    def __init__(self, n_estimators, n_stored_estimators, window_size, n_splits, base_estimator, random_state=None):
        super().__init__()

        # ensemble info
        self.max_member_count = n_estimators
        self.max_stored_count = n_stored_estimators
        if self.max_member_count > self.max_stored_count:
            self.max_stored_count = self.max_member_count
        self.ensemble = []
        self.ensemble_weights = []
        self.stored_learners = []
        self.stored_weights = []
        self.trainining_weight_seen_by_model = 0

        # misc
        self.n_splits = n_splits
        self.candidate_classifier = base_estimator
        self.class_distribution = []

        # chunk-related information
        self.window_size = window_size  # chunk size
        self.p = -1  # chunk pointer
        self.X_chunk = None
        self.y_chunk = None

        # set random state
        check_random_state(seed=random_state)

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def reset(self):
        self.X_chunk = None
        self.y_chunk = None
        self.ensemble = []
        self.stored_learners = []
        self.candidate_classifier.reset()
        self.p = -1

    def init_variables(self):
        pass

    def partial_fit(self, X, y, classes=None, weight=None):
        N, D = X.shape

        # initializes everything when the ensemble is first called
        if self.p == -1:
            self.X_chunk = np.zeros((self.window_size, D))
            self.y_chunk = np.zeros(self.window_size)
            self.p = 0

        # fill up the data chunk
        for i, x in enumerate(X):
            self.X_chunk[self.p] = X[i]
            self.y_chunk[self.p] = y[i]
            self.p += 1

            # update class distribution
            self.class_distribution[y[i]] += 1

            if self.p == self.window_size:
                self.process_chunk()

    def process_chunk(self):
        candidate_classifier_weight = self.compute_candidate_weight(self.candidate_classifier,
                                                                    self.X_chunk, self.y_chunk,
                                                                    self.n_splits)
        for i, learner in enumerate(self.stored_learners):
            self.stored_weights[i][0] = self.compute_weight(learner, self.X_chunk, self.y_chunk)

        if len(self.stored_learners) < self.max_stored_count:
            self.train_model(self.candidate_classifier, self.X_chunk, self.y_chunk)
            self.add_to_stored(self.candidate_classifier, candidate_classifier_weight)
        else:
            self.stored_weights.sort(key=lambda w: w[0], reverse=False)
            if self.stored_weights[0][0] < candidate_classifier_weight:
                self.train_model(self.candidate_classifier, self.X_chunk, self.y_chunk)
                self.stored_weights[0][0] = candidate_classifier_weight
                self.stored_learners[self.stored_weights[0][1]] = cp.deepcopy(self.candidate_classifier)

        ensemble_size = min(len(self.stored_learners), self.max_member_count)
        self.ensemble = [None] * ensemble_size
        self.ensemble_weights = np.zeros(ensemble_size)

        self.stored_weights.sort(key=lambda w: w[0])

        store_size = len(self.stored_learners)
        for i in range(len(ensemble_size)):
            self.ensemble_weights[i] = self.stored_weights[store_size - i - 1][0]
            self.ensemble[i] = self.stored_learners[self.stored_weights[store_size - i - 1][1]]

        self.class_distribution = None
        self.candidate_classifier.reset()

    def train_model(self, learner, X, y, classes=None, weight=None):
        try:
            learner.fit(X, y)
        except NotImplementedError:
            learner.partial_fit(X, y, classes, weight)

    def compute_candidate_weight(self, candidate, X, y, n_splits):
        candidate_weight = 0

        # get the k-fold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=0)
        for train_idx, test_idx in kf.split(X=self.X_chunk, y=self.y_chunk):
            X_train, y_train = self.X_chunk[train_idx], self.y_chunk[train_idx]
            X_test, y_test = self.X_chunk[test_idx], self.y_chunk[test_idx]
            learner = cp.deepcopy(self.candidate_classifier)
            self.train_model(learner, X_train, y_train)
            candidate_weight += self.compute_weight(learner, X_test, y_test)

        result_weight = candidate_weight / n_splits
        return result_weight

    def compute_weight(self, learner, X, y):
        mse_i = 0
        mse_r = 0

        votes_for_instance = learner.predict_proba(X)
        for i, inst in enumerate(X):
            vote_sum = 0
            votes_for_instance = votes_for_instance
            for element in votes_for_instance:
                vote_sum += element

            if vote_sum > 0:
                f_ci = votes_for_instance[i][y[i]] / vote_sum
                mse_i += (1 - f_ci) ** 2
            else:
                mse_i += 1

        mse_i = mse_i / self.window_size
        mse_r = self.compute_mse_r()
        return max(mse_r - mse_i, 0)

    def compute_mse_r(self):
        mse_r = 0
        for i, c in enumerate(self.class_distribution):
            p_c = c / self.window_size
            mse_r += p_c * ((1 - p_c) ** 2)
        return mse_r

    def add_to_stored(self, new_classifier, new_classifier_weight):
        added_classifier = None
        new_stored = [None] * (len(self.stored_learners) + 1)
        new_stored_weights = np.zeros((len(new_stored), 2))
        for i in range(len(new_stored)):
            if i < len(self.stored_learners):
                new_stored[i] = self.stored_learners[i]
                new_stored_weights[i][0] = self.stored_weights[i][0]
                new_stored_weights[i][1] = self.stored_weights[i][1]
            else:
                added_classifier = cp.deepcopy(new_classifier)
                new_stored[i] = added_classifier
                new_stored_weights[i][0] = new_classifier_weight
                new_stored_weights[i][1] = i
        self.stored_learners = new_stored
        self.stored_weights = new_stored_weights
        return added_classifier

    def discard_model(self, index):
        new_ensemble = [None] * (len(self.ensemble) + 1)
        new_ensemble_weights = np.zeros(len(new_ensemble))
        old_pos = 0
        for i in range(len(new_ensemble)):
            if old_pos == index:
                old_pos += 1
            new_ensemble[i] = self.ensemble[old_pos]
            new_ensemble_weights[i] = self.ensemble_weights[old_pos]
            old_pos += 1
        self.ensemble = new_ensemble
        self.ensemble_weights = new_ensemble_weights

    def predict(self, X):
        N, D = X.shape
        if len(self.ensemble) == 0:
            return np.zeros(N)

        probabs = self.predict_proba(X)
        predictions = np.zeros(N)
        for i, prob in enumerate(probabs):
            i_max = np.argmax(prob)
            predictions[i] = i_max
        return predictions

    def predict_proba(self, X):
        probabs = []
        N, D = X.shape
        L = len(self.class_distribution)

        if len(self.ensemble) == 0:
            return np.zeros((N, L))

        for i in range(N):
            combined_vote = []
            for i in range(len(self.ensemble)):
                if self.ensemble_weights[i] > 0.0:
                    vote = self.ensemble[i].predict_proba(X[i])
                    sum_vote = np.sum(vote)
                    if sum_vote > 0.0:
                        # normalize stuffs
                        vote = vote / sum_vote
                        vote = vote * (self.ensemble_weights[i] / (len(self.ensemble) + 1))
                        combined_vote.append(vote)
            # normalize combine_vote
            sum_combined_vote = np.sum(combined_vote)
            combined_vote = combined_vote / sum_combined_vote
            probabs.append(combined_vote)

        return probabs

    def score(self, X, y):
        raise NotImplementedError

    def get_info(self):
        """ Collects the information of the AWE

        Returns
        -------
        string
            Configuration for AWE.
        """

        description = type(self).__name__ + ': '
        description += "n_estimators: {} - ".format(self.max_member_count)
        description += "base_estimator: {} - ".format(type(self.max_stored_count))
        description += "window_size: {} - ".format(self.window_size)
        description += "n_splits: {} - ".format(self.n_splits)
        return description
