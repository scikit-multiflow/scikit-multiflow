import numpy as np
from skmultiflow.core.base import BaseEstimator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.utils.data_structures import InstanceWindow, FastBuffer, SlidingWindow

class Goowe(BaseEstimator):
    """ GOOWE (Geometrically Optimum Online Weighted Ensemble), as it is
    described in Bonab and Can (2017). Common notation in the code is
    as follows:
        K for maximum number of classifiers in the ensemble.
        N for data instances.
        A, d as they are, in the aforementioned paper.


    Parameters
    ----------
    n_max_components: int
        Ensemble size limit. Maximum number of component classifiers.
    chunk_size: int
        The amount of instances necessary for ensemble to learn concepts from.
        At each chunk_size many instances, some training is done.
    window_size: int
        Size of sliding window, which keeps record of the last k instances
        that are encountered in the data stream.
    """

    def __init__(self, n_max_components: int = 10,
                 chunk_size: int = 200, window_size: int = 100, logging = True):
        super().__init__()
        self._num_of_max_classifiers = n_max_components
        self._chunk_size = chunk_size
        self._Logging = logging
        self._num_of_current_classifiers = 0
        self._num_of_processed_instances = 0
        self._classifiers = np.empty((self._num_of_max_classifiers),
                                     dtype=object)
        self._weights = np.zeros((self._num_of_max_classifiers,))

        # What to save from current Data Chunk --> will be used for
        # adjusting weights, pruning purposes and so on.
        # Individual predictions of components, overall prediction of ensemble,
        # and ground truth info.
        self._chunk_comp_preds = FastBuffer(max_size=chunk_size)
        self._chunk_ensm_preds = FastBuffer(max_size=chunk_size)

        # chunk_data has instances in the chunk and their ground truth.
        # To be initialized after receiving n_features, n_targets
        self._chunk_data = None
        # self._chunk_truths = FastBuffer(max_size=chunk_size)

        # some external stuff that is about the data we are dealing with
        # but useful for recording predictions
        self._num_classes = None
        self._target_values = None      # Required to correctly train HTs
        self._record = False            # Boolean for keeping records to files

        # TODO: Implement Sliding Window Continuous Evaluator.
        # What to save at Sliding Window (last n instances) --> will be
        # used for continuous evaluation.
        # self._sliding_window_ensemble_preds =FastBuffer(max_size=window_size)
        # self._sliding_window_truths = FastBuffer(max_size=window_size)

    def prepare_post_analysis_req(self, num_features, num_targets, num_classes, target_values, record=False):
        # Need to get the dataset information but we do not want to
        # take it as an argument to the classifier itself, nor we do want to
        # ask it at each data instance. Hence we take dataset info from user
        # explicitly to create _chunk_data entries.
        #chunk_size = self._chunk_size
        '''
        self._chunk_data = InstanceWindow(n_features = num_features,
                                          n_targets = num_targets,
                                          max_size = self._chunk_size)
        '''
        self._chunk_data = SlidingWindow(window_size = self._chunk_size)
        #self._chunk_data = chunk_data
        # num_targets shows how many columns you want to predict in the data.
        # num classes is eqv to possible number of values that that column
        # can have.
        self._num_classes = num_classes
        self._target_values = target_values
        self._record = record

        if(self._record):
            # Create files that keeps record of:
            #   - weights at each chunk
            #   - individual component results for every instance
            #   - ground truths for every instance.
            self._f_comp_preds = open("component_predictions.csv", "w+")
            self._f_truths = open("ground_truths.csv", "w+")
            self._f_weights = open("weights.csv", "w+")

            self._f_comp_preds.write(str(self._chunk_size) + '\n')

            self._f_comp_preds.close()
            self._f_truths.close()
            self._f_weights.close()
        return

    def _get_components_predictions_for_instance(self, inst):
        """ For a given data instance, takes predictions of
        individual components from the ensemble as a matrix.

        Parameters
        ----------
        inst: data instance for which votes of components are delivered.

        Returns
        ----------
        numpy.array
            A 2-d numpy array where each row corresponds to predictions of
            each classifier.
        """
        preds = np.zeros((self._num_of_current_classifiers, self._num_classes))
        # print(np.shape(preds))
        for k in range(len(preds)):
            kth_comp_pred = self._classifiers[k].predict_proba(inst)
            # print(kth_comp_pred[0])
            # print(preds)
            # print("Component {}'s Prediction: {}".format(k, kth_comp_pred))
            preds[k, :] = kth_comp_pred[0]
        if(self._Logging):
            print('Component Predictions:')
            print(preds)
        return preds

    def _adjust_weights(self):
        """ Weight adustment by solving linear least squares, as it is
        described in Bonab and Can (2017).
        """
        # Prepare variables for Weight Adjustment
        # print('number of current classifiers: {}'.format(self._num_of_current_classifiers))
        A = np.zeros(shape=(self._num_of_current_classifiers,
                            self._num_of_current_classifiers))
        d = np.zeros(shape=(self._num_of_current_classifiers,))

        # Go over all the data chunk, calculate values of (S_i x S_j) for A.
        # (S_i x O) for d.
        #y_all = self._chunk_data.get_targets_matrix().astype(int)
        y_all = self._chunk_data.targets_buffer.astype(int)
        # print(y_all)
        for i in range(len(y_all)):
            class_index = y_all[i]
            comp_preds = self._chunk_comp_preds.get_next_element()
            #print("{} components predictions:".format(i))
            #print(comp_preds)

            A = A + comp_preds.dot(comp_preds.T)
            d = d + comp_preds[0][class_index]

        # A and d are filled. Now, the linear system Aw=d to be solved
        # to get our desired weights. w is of size K.
        # print("Solving Aw=d")
        # print(A)
        # print(d)
        w = np.linalg.lstsq(A, d, rcond=None)[0]

        # _weights has maximum size but what we found can be
        # smaller. Therefore, need to put the values of w to global weights
        if(self._num_of_current_classifiers < self._num_of_max_classifiers):
            for i in range(len(w)):
                self._weights[i] = w[i]
        else:                             # If full size, there is no problem.
            self._weights = w
        # print("After solving Aw=d weights:")
        # print(self._weights)
        return

    def _normalize_weights(self):
        """ Normalizes the weights of the ensemble to (0, 1) range.
        Performs (x_i - min(x)) / (max(x) - min(x)) on the nonzero elements
        of the weight vector.
        """
        min = np.amin(self._weights[:self._num_of_current_classifiers])
        max = np.amax(self._weights[:self._num_of_current_classifiers])

        if(min == max):     # all weights are the same
            for i in range(self._num_of_current_classifiers):
                self._weights[i] = 1. / self._num_of_current_classifiers
        else:
            for i in range(self._num_of_current_classifiers):
                self._weights[i] = (self._weights[i] - min) / (max - min)
        return

    def _normalize_weights_softmax(self):
        """ Normalizes the weights of the ensemble to (0, 1) range.
        Performs (x_i - min(x)) / (max(x) - min(x)) on the nonzero elements
        of the weight vector.
        """
        cur_weights = self._weights[:self._num_of_current_classifiers]
        self._weights[:self._num_of_current_classifiers] = np.exp(cur_weights) / sum(np.exp(cur_weights))

        return

    def _process_chunk(self):
        """ A subroutine that runs at the end of each chunk, allowing
        the components to be trained and ensemble weights to be adjusted.
        Until the first _process_chunk call, the ensemble is not yet ready.
        At first call, the first component is learned.
        At the rest of the calls, new components are formed, and the older ones
        are trained by the given chunk.
        If the ensemble size is reached, then the lowest weighted component is
        removed from the ensemble.
        """
        new_clf = HoeffdingTree()  # with default parameters for now
        new_clf.reset()

        # Save records of previous chunk
        if(self._record and self._num_of_current_classifiers > 0):
            self._record_truths_this_chunk()
            self._record_comp_preds_this_chunk()
            self._record_weights_this_chunk()

        # Case 1: No classifier in the ensemble yet, first chunk:
        if(self._num_of_current_classifiers == 0):
            self._classifiers[0] = new_clf
            self._weights[0] = 1.0  # weight is 1 for the first clf
            self._num_of_current_classifiers += 1
        else:
            # First, adjust the weights of the old component classifiers
            # according to what happened in this chunk.
            self._adjust_weights()
            # Case 2: There are classifiers in the ensemble but
            # the ensemble size is still not capped.
            if(self._num_of_current_classifiers < self._num_of_max_classifiers):
                # Put the new classifier to ensemble with the weight of 1
                self._classifiers[self._num_of_current_classifiers] = new_clf
                self._weights[self._num_of_current_classifiers] = float(1.0)
                self._num_of_current_classifiers += 1

            # Case 3: Ensemble size is capped. Need to replace the component
            # with lowest weight.
            else:
                assert (self._num_of_current_classifiers
                        == self._num_of_max_classifiers), "Ensemble not full."
                index_of_lowest_weight = np.argmin(self._weights)
                self._classifiers[index_of_lowest_weight] = new_clf
                self._weights[index_of_lowest_weight] = 1.0

            # Normalizing weigths to simplify numbers
            self._normalize_weights_softmax()       # maybe useful. we'll see.
            if(self._Logging):
                print("After normalization weights: ")
                print(self._weights)
        # Ensemble maintenance is done. Now train all classifiers
        # in the ensemble from the current chunk.
        # Can be parallelized.
        #data_features = self._chunk_data.get_attributes_matrix()
        data_features = self._chunk_data.features_buffer
        #data_truths = self._chunk_data.get_targets_matrix()
        data_truths = self._chunk_data.targets_buffer
        data_truths = data_truths.astype(int).flatten()

        if(self._Logging):
            print("Starting training the components with the current chunk...")
            for k in range(self._num_of_current_classifiers):
                print("Training classifier {}".format(k))
                self._classifiers[k].partial_fit(data_features, data_truths,
                                            classes=self._target_values)
            print("Training the components with the current chunk completed...")
        else:
            for k in range(self._num_of_current_classifiers):
                self._classifiers[k].partial_fit(data_features, data_truths, classes=self._target_values)
        return

    def _record_truths_this_chunk(self):
        f = open("ground_truths.csv", "ab")

        #data_truths = self._chunk_data.get_targets_matrix()
        data_truths = self._chunk_data.targets_buffer
        data_truths = data_truths.astype(int).flatten()

        # Default behaviour is to store list of lists for savetxt.
        # Hence, to prevent newline after each element of list, we surround
        # the truth array with one more set of bracketts.
        np.savetxt(f, [data_truths], delimiter=",", fmt='%d')

        f.close()
        return

    def _record_comp_preds_this_chunk(self):
        f = open("component_predictions.csv", "a+")
        np.savetxt(f, [self._num_of_current_classifiers], fmt='%d')

        comp_preds = np.array(self._chunk_comp_preds.get_queue())

        for i in range(len(comp_preds)):
            np.savetxt(f, comp_preds[i], delimiter=',', fmt='%1.5f')
        f.close()
        return

    def _record_weights_this_chunk(self):
        f = open("weights.csv", "a+")
        np.savetxt(f, [self._num_of_current_classifiers], fmt='%d')

        weights = self._weights
        np.savetxt(f, [weights], delimiter=',', fmt='%1.5f')
        f.close()
        return

    # --------------------------------------------------
    # Overridden methods from the parent (StreamModel)
    # --------------------------------------------------
    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError("For now, only the stream version "
                                  "is implemented. Use partial_fit()")

    def partial_fit(self, X, y, classes=None, weight=None):
        # This method should work with individual instances, as well as bunch
        # of instances, since there can be pre-training for warm start.

        # If an individual instance is inputted, then just save X and y to
        # train from them later.
        if(len(X) == 1):
            # Save X and y to train classifiers later
            # y is required to be 1x1, and hence the square bracketts.
            y_i = np.array([y])
            # print(type(X))
            # print(type(y_i))
            # print(X)
            # print(y_i)
            self._chunk_data.add_sample(X, y_i)

            # If still filling the chunk, then just add the instance to the
            # current data chunk, wait for it to be filled.
            self._num_of_processed_instances += 1

            # If at the end of a chunk, start training components
            # and adjusting weights using information in this chunk.
            if(self._num_of_processed_instances % self._chunk_size == 0):
                print("Instance {}".format(self._num_of_processed_instances))
                self._process_chunk()
        elif(len(X) > 1):
            # Input is a chunk. Add them individually.
            for i in range(len(X)):
                X_i = np.array([X[i]])
                y_i = np.array([[y[i]]])
                # print(X_i)
                # print(y_i)
                self._chunk_data.add_sample(X_i, y_i)
                self._num_of_processed_instances += 1

                # If at the end of a chunk, start training components
                # and adjusting weights using information in this chunk.
                if(self._num_of_processed_instances % self._chunk_size == 0):
                    print("Instance {}".format(self._num_of_processed_instances))
                    self._process_chunk()
        else:
            print("Something wrong with the data...")
            print("len(X) is: {}".format(len(X)))
        return

    def predict(self, X):
        """ For a given data instance, yields the prediction values.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.
        """
        predictions = []
        if(len(X) == 1):
            predictions.append(np.argmax(self.predict_proba(X)))
        elif(len(X) > 1):
            # Add many predictions
            for i in range(len(X)):
                relevance_scores = self.predict_proba(X[i])
                predictions.append(np.argmax(relevance_scores))
        # print(np.argmax(relevance_scores))
        if(self._Logging):
            print('Ensemble Prediction:')
            print(np.array(predictions))
        return np.array(predictions) #, one_hot

    def predict_proba(self, X):
        """ For a given data instance, takes WEIGHTED combination
        of components to get relevance scores for each class.

        Parameters
        ----------
        X: data instance for which weighted combination is delivered.

        Returns
        ----------
        numpy.array
            A vector with number_of_classes elements where each element
            represents class score of corresponding class for this instance.
        """
        weights = np.array(self._weights)

        # get only the useful weights
        weights = weights[:self._num_of_current_classifiers]
        components_preds = self._get_components_predictions_for_instance(X)
        #print('*****************************')
        #print(components_preds)
        #print('*****************************')
        # Save individual component predictions and ensemble prediction
        # for later analysis.
        self._chunk_comp_preds.add_element([components_preds])

        #print(weights)
        #print(components_preds)
        #print(self.get_classifiers())
        weighted_ensemble_vote = np.dot(weights, components_preds)
        # print("Weighted Ensemble vote: {}".format(weighted_ensemble_vote))
        self._chunk_ensm_preds.add_element(weighted_ensemble_vote)

        return weighted_ensemble_vote

    def reset(self):
        pass

    def score(self, X, y):
        pass

    def get_info(self):
        return 'The Ensemble GOOWE (Bonab and Can, 2017) with' + \
            ' - n_max_components: ' + str(self._num_of_max_classifiers) + \
            ' - num_of_current_components: ' + str(self._num_of_current_classifiers) + \
            ' - chunk_size: ' + str(self._chunk_size) + \
            ' - num_dimensions_in_label_space(num_classes): ' + str(self._num_classes) + \
            ' - recording: ' + str(self._record)

    def get_class_type(self):
        pass

    # Some getters and setters..
    def get_number_of_current_classifiers(self):
        return self._num_of_current_classifiers

    def get_number_of_max_classifiers(self):
        return self._num_of_max_classifiers

    # Helper methods for GooweMS
    def get_classifiers(self):
        return self._classifiers

    def set_classifiers(self, classifiers):
        self._classifiers = classifiers

    def get_weights(self):
        return self._weights
