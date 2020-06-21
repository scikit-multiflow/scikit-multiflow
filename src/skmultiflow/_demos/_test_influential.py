import operator

from skmultiflow.data import influential_stream
from skmultiflow.evaluation import evaluate_influential
from skmultiflow.trees import HoeffdingTreeClassifier


def demo():
    """ _test_influential

    This demo tests if the streams are correctly created and
    if the classifier chooses a new sample based on the weights
    of the streams.

    :return:
    """

    stream = influential_stream.InfluentialStream(self_defeating=0.99, self_fulfilling=1.01)

    ht = HoeffdingTreeClassifier()

    # 3. Setup the evaluator
    evaluator = evaluate_influential.EvaluateInfluential(show_plot=True,
                                                         pretrain_size=200,
                                                         max_samples=20000,
                                                         batch_size=1,
                                                         time_windows=2,
                                                         intervals=4,
                                                         metrics=['accuracy'],
                                                         data_points_for_classification=True)

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=ht)


if __name__ == '__main__':
    demo()
