import operator

from skmultiflow.data import influential_stream
from skmultiflow.evaluation import evaluate_influential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import naive_bayes
from skmultiflow.core import Pipeline


def demo():
    """ _test_influential

    This demo tests if the streams are correctly created and
    if the classifier chooses a new sample based on the weights
    of the streams.

    :return:
    """

    stream = influential_stream.InfluentialStream(self_defeating=0.9999, self_fulfilling=1.0001)

    classifier = naive_bayes.NaiveBayes()
    # classifier = PerceptronMask()
    # classifier = HoeffdingTreeClassifier()
    # classifier = PassiveAggressiveClassifier()

    # 3. Setup the evaluator
    evaluator = evaluate_influential.EvaluateInfluential(show_plot=False,
                                                         pretrain_size=200,
                                                         max_samples=20000,
                                                         batch_size=1,
                                                         n_time_windows=2,
                                                         n_intervals=3,
                                                         metrics=['accuracy'],
                                                         data_points_for_classification=False,
                                                         track_weight=False)

    pipe = Pipeline([('Naive Bayes', classifier)])

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=pipe)


if __name__ == '__main__':
    demo()
