from copy import deepcopy
from skmultiflow.core.base_object import BaseObject
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.trees.hoeffding_tree import *
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees.arf_hoeffding_tree import ARFHoeffdingTree
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from skmultiflow.utils import check_random_state, check_weights


class AdaptiveRandomForest(StreamModel):
    """Adaptive Random Forest (ARF).

        Parameters
        ----------
        n_estimators: int, optional (default=10)
            Number of trees oin the ensemble.

        max_features : int, float, string or None, optional (default="auto")
            Max number of attributes for each node split.

            - If int, then consider ``max_features`` features at each split.
            - If float, then ``max_features`` is a percentage and ``int(max_features * n_features)``
              features are considered at each split.
            - If "auto", then ``max_features=sqrt(n_features)``.
            - If "sqrt", then ``max_features=sqrt(n_features)`` (same as "auto").
            - If "log2", then ``max_features=log2(n_features)``.
            - If None, then ``max_features=n_features``.
        disable_weighted_vote: bool, optional (default=False)
            Weighted vote option.

        lambda_value: int, optional (default=6)
            The lambda value for bagging (lambda=6 corresponds to Leverage Bagging).

        performance_metric: string, optional (default="acc")
            Metric used to track trees performance within the ensemble.

            - 'acc' - Accuracy
            - 'kappa' - Accuracy
        drift_detection_method: BaseDriftDetector or None,, optional (default=ADWIN(0.001))
            Drift Detection method. Set to None to disable Drift detection.

        warning_detection_method: BaseDriftDetector or None, default(ADWIN(0.01))
            Warning Detection method. Set to None to disable warning detection.

        max_byte_size: int, optional (default=33554432)
            (`ARFHoeffdingTree` parameter)
            Maximum memory consumed by the tree.

        memory_estimate_period: int, optional (default=2000000)
            (`ARFHoeffdingTree` parameter)
            Number of instances between memory consumption checks.

        grace_period: int, optional (default=50)
            (`ARFHoeffdingTree` parameter)
            Number of instances a leaf should observe between split attempts.

        split_criterion: string, optional (default='info_gain')
            (`ARFHoeffdingTree` parameter)
            Split criterion to use.

            - 'gini' - Gini
            - 'info_gain' - Information Gain

        split_confidence: float, optional (default=0.01)
            (`ARFHoeffdingTree` parameter)
            Allowed error in split decision, a value closer to 0 takes longer to decide.

        tie_threshold: float, optional (default=0.05)
            (`ARFHoeffdingTree` parameter)
            Threshold below which a split will be forced to break ties.

        binary_split: bool, optional (default=False)
            (`ARFHoeffdingTree` parameter)
            If True, only allow binary splits.

        stop_mem_management: bool, optional (default=False)
            (`ARFHoeffdingTree` parameter)
            If True, stop growing as soon as memory limit is hit.

        remove_poor_atts: bool, optional (default=False)
            (`ARFHoeffdingTree` parameter)
            If True, disable poor attributes.

        no_preprune: bool, optional (default=False)
            (`ARFHoeffdingTree` parameter)
            If True, disable pre-pruning.

        leaf_prediction: string, optional (default='nba')
            (`ARFHoeffdingTree` parameter)
            Prediction mechanism used at leafs.

            - 'mc' - Majority Class
            - 'nb' - Naive Bayes
            - 'nba' - Naive Bayes Adaptive

        nb_threshold: int, optional (default=0)
            (`ARFHoeffdingTree` parameter)
            Number of instances a leaf should observe before allowing Naive Bayes.

        nominal_attributes: list, optional
            (`ARFHoeffdingTree` parameter)
            List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

        random_state: int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by `np.random`.


        Notes
        -----
        The 3 most important aspects of Adaptive Random Forest [1]_ are:
        (1) inducing diversity through re-sampling;
        (2) inducing diversity through randomly selecting subsets of features for node splits (see
        skmultiflow.classification.trees.arf_hoeffding_tree);
        (3) drift detectors per base tree, which cause selective resets in response to drifts.
        It also allows training background trees, which start training if a warning is detected and replace the active
        tree if the warning escalates to a drift.

        References
        ----------
        .. [1] Heitor Murilo Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal, Fabricio Enembreck,
           Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem.
           Adaptive random forests for evolving data stream classification.
           In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.

    """

    def __init__(self,
                 n_estimators=10,
                 max_features='auto',
                 disable_weighted_vote=False,
                 lambda_value=6,
                 performance_metric='acc',
                 drift_detection_method: BaseDriftDetector=ADWIN(0.001),
                 warning_detection_method: BaseDriftDetector=ADWIN(0.01),
                 max_byte_size=33554432,
                 memory_estimate_period=2000000,
                 grace_period=50,
                 split_criterion='info_gain',
                 split_confidence=0.01,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 random_state=None):
        """AdaptiveRandomForest class constructor."""
        super().__init__()          
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.disable_weighted_vote = disable_weighted_vote
        self.lambda_value = lambda_value
        if isinstance(drift_detection_method, BaseDriftDetector):
            self.drift_detection_method = drift_detection_method
        else:
            self.drift_detection_method = None
        if isinstance(warning_detection_method, BaseDriftDetector):
            self.warning_detection_method = warning_detection_method
        else:
            self.warning_detection_method = None
        self.instances_seen = 0
        self._train_weight_seen_by_model = 0.0
        self.ensemble = None
        self._init_random_state = random_state
        self.random_state = check_random_state(self._init_random_state)
        if performance_metric in ['acc', 'kappa']:
            self.performance_metric = performance_metric
        else:
            raise ValueError('Invalid performance metric: {}'.format(performance_metric))

        # ARH Hoeffding Tree configuration
        self.max_byte_size = max_byte_size
        self. memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError
    
    def partial_fit(self, X, y, classes=None, weight=1.0):
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            weight = check_weights(weight, expand_length=row_cnt)
            for i in range(row_cnt):
                if weight[i] != 0.0:
                    self._train_weight_seen_by_model += weight[i]
                    self._partial_fit(X[i], y[i], weight[i])
        
    def _partial_fit(self, X, y, weight):
        self.instances_seen += 1
        
        if self.ensemble is None:
            self.init_ensemble(X)

        for i in range(self.n_estimators):
            y_predicted = self.ensemble[i].predict(np.asarray([X]))
            self.ensemble[i].evaluator.add_result(y_predicted, y, weight)
            k = self.random_state.poisson(self.lambda_value)
            if k > 0:
                self.ensemble[i].partial_fit(np.asarray([X]), np.asarray([y]), np.asarray([k]), self.instances_seen)
    
    def predict(self, X):
        """Predicts the label of the X instance(s)
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.
        Returns
        -------
        list
            Predicted labels for all instances in X.
        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = self.get_votes_for_instance(X[i])
            if votes == {}:
                # Ensemble is empty, all classes equal, default to zero
                predictions.append(0)
            else:
                predictions.append(max(votes, key=votes.get))
        return predictions

    def predict_proba(self, X):
        raise NotImplementedError
        
    def reset(self):        
        """Reset ARF."""
        self.ensemble = None
        self.max_features = 0
        self.instances_seen = 0
        self._train_weight_seen_by_model = 0.0
        self.random_state = check_random_state(self._init_random_state)

    def score(self, X, y):
        raise NotImplementedError
        
    def get_info(self):
        """Collect information about the Hoeffding Tree configuration.

        Returns
        -------
        string
            Configuration for the Hoeffding Tree.
        """
        description = type(self).__name__ + ': '
        description += 'n_estimators: {} - '.format(self.n_estimators)
        description += 'max_features: {} - '.format(self.max_features)
        description += 'disable_weighted_vote: {} - '.format(self.disable_weighted_vote)
        description += 'lambda_value: {} - '.format(self.lambda_value)
        return description

    def get_votes_for_instance(self, X):
        if self.ensemble is None:
            self.init_ensemble(X)
        combined_votes = {}

        for i in range(self.n_estimators):
            vote = self.ensemble[i].get_votes_for_instance(X)
            if vote != {} and sum(vote.values()) > 0:
                normalize_values_in_dict(vote)
                if not self.disable_weighted_vote:
                    performance = self.ensemble[i].evaluator.get_performance()\
                        if self.performance_metric == 'acc'\
                        else self.ensemble[i].evaluator.get_kappa()
                    if performance != 0.0:  # CHECK How to handle negative (kappa) values?
                        for k in vote:
                            vote[k] = vote[k] * performance
                # Add values
                for k in vote:
                    try:
                        combined_votes[k] += vote[k]
                    except KeyError:
                        combined_votes[k] = vote[k]
        return combined_votes
        
    def init_ensemble(self, X):
        self.ensemble = [None] * self.n_estimators

        self._set_max_features(get_dimensions(X)[1])

        for i in range(self.n_estimators):
            self.ensemble[i] = ARFBaseLearner(i,
                                              ARFHoeffdingTree(max_byte_size=self.max_byte_size,
                                                               memory_estimate_period=self.memory_estimate_period,
                                                               grace_period=self.grace_period,
                                                               split_criterion=self.split_criterion,
                                                               split_confidence=self.split_confidence,
                                                               tie_threshold=self.tie_threshold,
                                                               binary_split=self.binary_split,
                                                               stop_mem_management=self.stop_mem_management,
                                                               remove_poor_atts=self.remove_poor_atts,
                                                               no_preprune=self.no_preprune,
                                                               leaf_prediction=self.leaf_prediction,
                                                               nb_threshold=self.nb_threshold,
                                                               nominal_attributes=self.nominal_attributes,
                                                               max_features=self.max_features,
                                                               random_state=self._init_random_state),
                                              self.instances_seen,
                                              self.drift_detection_method,
                                              self.warning_detection_method,
                                              False)

    def _set_max_features(self, n):
        if self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features = round(math.sqrt(n))
        elif self.max_features == 'log2':
            self.max_features = round(math.log2(n))
        elif isinstance(self.max_features, int):
            # Consider 'max_features' features at each split.
            pass
        elif isinstance(self.max_features, float):
            # Consider 'max_features' as a percentage
            self.max_features = int(self.max_features * n)
        elif self.max_features is None:
            self.max_features = n
        else:
            # Default to "auto"
            self.max_features = round(math.sqrt(n))
        # Sanity checks
        # max_features is negative, use max_features + n
        if self.max_features < 0:
            self.max_features += n
        # max_features <= 0 (m can be negative if max_features is negative and abs(max_features) > n),
        # use max_features = 1
        if self.max_features <= 0:
            self.max_features = 1
        # max_features > n, then use n
        if self.max_features > n:
            self.max_features = n

    @staticmethod
    def is_randomizable():
        return True                  


class ARFBaseLearner(BaseObject):
    """ARF Base Learner class.

    Parameters
    ----------
    index_original: int
        Tree index within the ensemble.
    classifier: ARFHoeffdingTree
        Tree classifier.
    instances_seen: int
        Number of instances seen by the tree.
    drift_detection_method: BaseDriftDetector
        Drift Detection method.
    warning_detection_method: BaseDriftDetector
        Warning Detection method.
    is_background_learner: bool
        True if the tree is a background learner.

    Notes
    -----
    Inner class that represents a single tree member of the forest.
    Contains analysis information, such as the numberOfDriftsDetected.

    """
    def __init__(self,
                 index_original,
                 classifier: ARFHoeffdingTree,
                 instances_seen,
                 drift_detection_method: BaseDriftDetector,
                 warning_detection_method: BaseDriftDetector,
                 is_background_learner):
        self.index_original = index_original
        self.classifier = classifier 
        self.created_on = instances_seen
        self.is_background_learner = is_background_learner
        self.evaluator_method = ClassificationMeasurements

        # Drift and warning
        self.drift_detection_method = drift_detection_method
        self.warning_detection_method = warning_detection_method

        self.last_drift_on = 0
        self.last_warning_on = 0
        self.nb_drifts_detected = 0
        self.nb_warnings_detected = 0            

        self.drift_detection = None
        self.warning_detection = None
        self.background_learner = None
        self._use_drift_detector = False
        self._use_background_learner = False
        
        self.evaluator = self.evaluator_method()

        # Initialize drift and warning detectors
        if drift_detection_method is not None:
            self._use_drift_detector = True
            self.drift_detection = deepcopy(drift_detection_method)

        if warning_detection_method is not None:
            self._use_background_learner = True
            self.warning_detection = deepcopy(warning_detection_method)
            
    def reset(self, instances_seen):
        if self._use_background_learner and self.background_learner is not None:
            self.classifier = self.background_learner.classifier
            self.warning_detection = self.background_learner.warning_detection
            self.drift_detection = self.background_learner.drift_detection
            self.evaluator_method = self.background_learner.evaluator_method
            self.created_on = self.background_learner.created_on                
            self.background_learner = None
        else:
            self.classifier.reset()
            self.created_on = instances_seen
            self.drift_detection.reset()
        self.evaluator = self.evaluator_method()

    def partial_fit(self, X, y, weight, instances_seen):
        self.classifier.partial_fit(X, y, weight)

        if self.background_learner:
            self.background_learner.classifier.partial_fit(X, y, weight)

        correctly_classifies = False
        if self._use_drift_detector and not self.is_background_learner:
            correctly_classifies = self.classifier.predict(X) == y
            # Check for warning only if use_background_learner is active
            if self._use_background_learner:
                self.warning_detection.add_element(int(not correctly_classifies))
                # Check if there was a change
                if self.warning_detection.detected_change():
                    self.last_warning_on = instances_seen
                    self.nb_warnings_detected += 1
                    # Create a new background tree classifier
                    background_learner = self.classifier.new_instance()
                    # Create a new background learner object
                    self.background_learner = ARFBaseLearner(self.index_original,
                                                             background_learner,
                                                             instances_seen,
                                                             self.drift_detection_method,
                                                             self.warning_detection_method,
                                                             True)
                    # Update the warning detection object for the current object
                    # (this effectively resets changes made to the object while it was still a bkg learner).
                    self.warning_detection.reset()

        # Update the drift detection
        self.drift_detection.add_element(int(not correctly_classifies))

        # Check if there was a change
        if self.drift_detection.detected_change():
            self.last_drift_on = instances_seen
            self.nb_drifts_detected += 1
            self.reset(instances_seen)

    def predict(self, X):
        return self.classifier.predict(X)
    
    def get_votes_for_instance(self, X):
        return self.classifier.get_votes_for_instance(X)

    def get_class_type(self):
        raise NotImplementedError

    def get_info(self):
        return "NotImplementedError"
