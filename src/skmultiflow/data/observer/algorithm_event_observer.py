from skmultiflow.data.observer.event_observer import EventObserver

#TODO: add
# - strategies to compute features on the fly
# - evaluation strategy: holdout vs prequential
# - results observer, supporting async results reporting + policies to discard data that arrives too late?
class AlgorithmEventObserver(EventObserver):
    """ AlgorithmEventObserver class.

    This abstract class defines the minimum requirements of a data source.
    It provides an interface to access entries, streamed from certain source.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    def update(self, event):
        algorithm_type = self.algorithm.algorithm_type()
        X = event['X']
        if algorithm_type == 'CLASSIFICATION' or algorithm_type == 'REGRESSION':
            y = event['y']

        y_pred = self.algorithm.predict(X)

        >> > if y[0] == y_pred[0]:
            >> > correct_cnt += 1
        >> > ht = ht.partial_fit(X, y)
