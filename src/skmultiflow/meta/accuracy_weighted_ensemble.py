from skmultiflow.core.base import StreamModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, KFold
import sortedcontainers as sc
import numpy as np
import copy as cp
import operator


class AccuracyWeightedEnsemble(StreamModel):
    """
    Accuracy Weighted Ensemble or AWE

    Parameters
    ----------
    n_estimators: int (default=10)
        Maximum number of estimators to be kept in the ensemble
    base_estimator: StreamModel or sklearn.BaseEstimator (default=DecisionTreeClassifier)
        Each member of the ensemble is an instance of the base estimator
    window_size: int (default=200)
        The size of one chunk to be processed
        (warning: the chunk size is not always the same as the batch size)
    n_splits: int (default=5)
        Number of folds to run cross-validation for computing the weight
        of a classifier in the ensemble

    Notes
    -----
    An Accuracy Weighted Ensemble (AWE) [1]_ is an ensemble of classification models in which
    each model is judiciously weighted based on their expected classification accuracy
    on the test data under the time-evolving environment. The ensemble guarantees to be
    efficient and robust against concept-drifting streams.

    References
    ----------
    .. [1] Haixun Wang, Wei Fan, Philip S. Yu, and Jiawei Han. 2003.
    Mining concept-drifting data streams using ensemble classifiers.
    In Proceedings of the ninth ACM SIGKDD international conference
    on Knowledge discovery and data mining (KDD '03).
    ACM, New York, NY, USA, 226-235.

    """

    class WeightedClassifier:
        """ A wrapper that includes a base estimator and its associated weight
        (and additional information)

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The base estimator to be wrapped up with additional information.
            This estimator must already been trained on a data chunk.
        weight: float
            The weight associated to this estimator
        chunk_labels: array
            The array containing the unique class labels of the data chunk this estimator
            is trained on.
        """

        def __init__(self, estimator, weight, chunk_labels):
            """ Creates a new weighted classifier."""
            self.estimator = estimator
            self.weight = weight
            self.chunk_labels = chunk_labels

        def __lt__(self, other):
            """ Compares an object of this class to the other by means of the weight.
            This method helps to sort the classifier correctly in the sorted list.

            Parameters
            ----------
            other: WeightedClassifier
                The other object to be compared to

            Returns
            -------
            boolean
                true if this object's weight is less than that of the other object
            """
            return self.weight < other.weight

    def __init__(self, n_estimators=10, n_kept_estimators=30,
                 base_estimator=DecisionTreeClassifier(), window_size=200, n_splits=5):
        """ Create a new ensemble"""

        super().__init__()

        # top K classifiers
        self.n_estimators = n_estimators

        # total number of classifiers to keep
        self.n_kept_estimators = n_kept_estimators

        # base learner
        self.base_estimator = base_estimator

        # the ensemble in which the classifiers are sorted by their weight
        # self.models = sc.SortedList()
        self.models_pool = sc.SortedList()

        # cross validation fold
        self.n_splits = n_splits

        # chunk-related information
        self.window_size = window_size  # chunk size
        self.p = -1  # chunk pointer
        self.X_chunk = None
        self.y_chunk = None

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y=None, classes=None, weight=None):
        """ Updates the ensemble when a new data chunk arrives (Algorithm 1 in the paper).
        The update is only launched when the chunk is filled up.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.
        """

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

            if self.p == self.window_size:
                # reset the pointer
                self.p = 0

                # retrieve the classes and class count
                if classes is None:
                    classes, class_count = np.unique(self.y_chunk, return_counts=True)
                else:
                    _, class_count = np.unique(self.y_chunk, return_counts=True)

                # (1) train classifier C' from X by creating a deep copy from the base learner
                C_new = cp.deepcopy(self.base_estimator)
                try:
                    C_new.fit(self.X_chunk, self.y_chunk)
                except NotImplementedError:
                    C_new.partial_fit(self.X_chunk, self.y_chunk, classes, weight)

                # compute the baseline error rate given by a random classifier
                baseline_score = self.compute_baseline(classes)

                # compute the weight of C', may do cross-validation if cv is not None
                clf_new = self.WeightedClassifier(estimator=C_new, weight=0, chunk_labels=classes)
                clf_new.weight = self.compute_weight(model=clf_new, baseline_score=baseline_score,
                                                     n_splits=self.n_splits)

                # (4) update the weights of each classifier in the ensemble, not using cross-validation
                # for model in self.models:
                #     model.weight = self.compute_weight(model=model, baseline_score=baseline_score, n_splits=None)
                for model in self.models_pool:
                    model.weight = self.compute_weight(model=model, baseline_score=baseline_score, n_splits=None)

                # (5) C <- top K weighted classifiers in C U { C' }
                # if len(self.models) < self.n_estimators:
                #     self.models.add(value=clf_new)
                # else:
                #     if clf_new.weight > 0 and clf_new.weight > self.models[0].weight:
                #         self.models.pop(0)
                #         self.models.add(value=clf_new)

                # add the new model to the pool if there are slots available
                if len(self.models_pool) < self.n_kept_estimators:
                    self.models_pool.add(clf_new)
                else:
                    # remove the worst one
                    if clf_new.weight > 0 and clf_new.weight > self.models[0].weight:
                        self.models_pool.pop(0)
                        self.models_pool.add(clf_new)

                # instance-based pruning only happens with Cost Sensitive extension
                self.do_instance_pruning()

        return self

    def do_instance_pruning(self):
        # only has effect if the ensemble is applied in cost-sensitive applications
        pass

    def predict(self, X):
        """ Predicts the labels of X in a general classification setting.
        The prediction is done via normalized weighted voting (choosing the maximum).

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.
        """

        N, D = X.shape
        # K = self.n_estimators if len(self.models_pool) >= self.n_estimators else len(self.models_pool)
        ensemble = list(self.models_pool.islice(len(self.models_pool) - self.n_estimators - 1, len(self.models_pool)))
        sum_weights = np.sum([clf.weight for clf in ensemble])  # for normalization

        # if using array
        # predictions = np.zeros(N)
        # predictions_by_all_models = np.column_stack([model.estimator.predict(X) for model in self.models])
        # for i, pred in enumerate(predictions_by_all_models):
        #     avg_weighted_prediction = {}
        #     for j, label in enumerate(pred):
        #         if label in avg_weighted_prediction:
        #             avg_weighted_prediction[label] += self.models[j].weight / sum_weights
        #         else:
        #             avg_weighted_prediction[label] = self.models[j].weight / sum_weights
        #     max_value = max(avg_weighted_prediction.items(), key=operator.itemgetter(1))[0]
        #     predictions[i] = max_value
        # return predictions

        weigthed_votes = [dict()] * N
        for model in ensemble:
            classifier = model.estimator
            prediction = classifier.predict(X)
            weight = model.weight
            for i, label in enumerate(prediction):
                try:
                    weigthed_votes[i][label] += weight / sum_weights
                except KeyError:
                    weigthed_votes[i][label] = weight / sum_weights

        predict_weighted_voting = np.zeros(N)
        for i, dic in enumerate(weigthed_votes):
            max_value = max(dic.items(), key=operator.itemgetter(1))[0]
            predict_weighted_voting[i] = max_value

        return predict_weighted_voting

    def predict_proba(self, X):
        raise NotImplementedError

    def reset(self):
        """ Resets all parameters to its default value"""
        self.n_estimators = 10
        self.base_estimator = DecisionTreeClassifier()
        self.models_pool = sc.SortedList()
        self.n_splits = 5

        # chunk-related information
        self.window_size = 200
        self.p = -1
        self.X_chunk = None
        self.y_chunk = None

    def score(self, X, y):
        raise NotImplementedError

    def compute_score(self, model, X, y):
        """ Computes the mean square error of a classifier, via the predicted probabilities.

        This code needs to take into account the fact that a classifier C trained on a
        previous data chunk may not have seen all the labels that appear in a new chunk
        (e.g. C is trained with only labels [1, 2] but the new chunk contains labels [1, 2, 3, 4, 5]

        Parameters
        ----------
        model: StreamModel or sklearn.BaseEstimator
            The estimator in the ensemble to compute the score on
        X: numpy.ndarray of shape (window_size, n_features)
            The data from the new chunk
        y: numpy.array
            The labels from the new chunk

        Returns
        -------
        float
            The mean square error of the model (MSE_i)
        """
        N = len(y)
        labels = model.chunk_labels
        probabs = model.estimator.predict_proba(X)
        sum_error = 0
        for i, c in enumerate(y):
            # if the label in y is unseen when training,
            # skip it, don't include it in the error
            if c in labels:
                index_label_c = np.where(labels == c)[0][0]  # find the index of this label c in probabs[i]
                probab_ic = probabs[i][index_label_c]
                sum_error += (1 - probab_ic) ** 2
            else:
                sum_error += 1

        return sum_error / N

    def compute_score_crossvalidation(self, model, n_splits):
        """ Computes the score of interests, using cross-validation or not.

        Parameters
        ----------
        model: StreamModel or sklearn.BaseEstimator
            The estimator in the ensemble to compute the score on
        n_splits: int
            The number of CV folds.
            If None, the score is computed directly on the entire data chunk.
            Else, we proceed as in traditional cross-validation setting.

        Returns
        -------
        float
            The score of an estimator computed via CV
        """

        if n_splits is not None and type(n_splits) is int:
            # we create a copy because we don't want to "modify" an already trained model
            copy_model = cp.deepcopy(model)
            copy_model.estimator = cp.deepcopy(self.base_estimator) # make a new estimator
            score = 0
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=0)
            for train_idx, test_idx in kf.split(X=self.X_chunk, y=self.y_chunk):
                X_train, y_train = self.X_chunk[train_idx], self.y_chunk[train_idx]
                X_test, y_test = self.X_chunk[test_idx], self.y_chunk[test_idx]
                try:
                    copy_model.estimator.fit(X_train, y_train)
                except NotImplementedError:
                    copy_model.estimator.partial_fit(X_train, y_train, copy_model.chunk_labels, None)
                score += self.compute_score(model=copy_model, X=X_test, y=y_test) / self.n_splits
        else:
            # compute the score on the entire data chunk
            score = self.compute_score(X=self.X_chunk, y=self.y_chunk, model=model)

        return score

    def compute_weight(self, model, baseline_score, n_splits=None):
        """ Computes the weight of a classifier given the baseline score calculated on a random learner.
        The weight relies on either (1) MSE if it is a normal classifier,
        or (2) benefit if it is a cost-sensitive classifier.

        Parameters
        ----------
        model: StreamModel or sklearn.BaseEstimator
            The learner to compute the weight on
        baseline_score: float
            The baseline score calculated on a random learner
        n_splits: int (default=None)
            The number of CV folds.
            If not None (and is a number), we compute the weight using CV

        Returns
        -------
        float
            The weight computed from the MSE score of the classifier
        """

        # compute MSE, with cross-validation or not
        score = self.compute_score_crossvalidation(model=model, n_splits=n_splits)

        # w = MSE_r = MSE_i
        return baseline_score - score

    def compute_baseline(self, classes):
        """ This method computes the score produced by a random classifier, served as a baseline.
        The baseline score is MSE_r in case of a normal classifier, b_r in case of a cost-sensitive one

        Parameters
        ----------
        classes: numpy.array
            The unique class labels

        Returns
        -------
        float
            The baseline score of a random learner
        """

        # if we assume uniform distribution
        # L = len(np.unique(classes))
        # MSE_r = L * (1 / L) * (1 - 1 / L) ** 2

        # if we base on the class distribution of the data --> count the number of labels
        _, class_count = np.unique(classes, return_counts=True)
        class_dist = [class_count[i] / self.window_size for i, c in enumerate(classes)]
        mse_r = np.sum([class_dist[i] * ((1 - class_dist[i]) ** 2) for i, c in enumerate(classes)])
        return mse_r

    def get_info(self):
        """ Collects the information of the AWE

        Returns
        -------
        string
            Configuration for AWE.
        """

        description = type(self).__name__ + ': '
        description += "n_estimators: {} - ".format(self.n_estimators)
        description += "base_estimator: {} - ".format(type(self.base_estimator))
        description += "window_size: {} - ".format(self.window_size)
        description += "n_splits: {} - ".format(self.n_splits)
        return description
