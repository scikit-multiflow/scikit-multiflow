import operator

from skmultiflow.data import influential_stream, random_rbf_generator
from skmultiflow.evaluation import evaluate_influential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import naive_bayes
from skmultiflow.core import Pipeline
from prettytable import PrettyTable
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift


def demo():
    """ _test_influential

    This demo tests if the streams are correctly created and
    if the classifier chooses a new sample based on the weights
    of the streams.

    :return:
    """
    equal = PrettyTable()
    equal.field_names = ["Run", "Feature number", "False Negative", "True Positive",
                         "True Negative", "False positive", "p-value"]
    for i in range(50):
        stream = influential_stream.InfluentialStream(self_defeating=1, self_fulfilling=1.001,
                                                      streams=[RandomRBFGenerator(model_random_state=99,
                                                                                  sample_random_state=50,
                                                                                  n_classes=2, n_features=2,
                                                                                  n_centroids=50),
                                                               RandomRBFGeneratorDrift(model_random_state=112,
                                                                                       sample_random_state=50,
                                                                                       n_classes=2,
                                                                                       n_features=2,
                                                                                       n_centroids=(i+1)*2,
                                                                                       change_speed=0.9,
                                                                                       num_drift_centroids=i*2)])
        evaluating(stream, i, equal)
    #
    # fulfilling = PrettyTable()
    # fulfilling.field_names = ["Run", "Feature number", "False Negative", "True Positive",
    #                      "True Negative", "False positive", "p-value"]
    # for i in range(100):
    #     stream = influential_stream.InfluentialStream(self_defeating=1, self_fulfilling=1.001)
    #     evaluating(stream, i, fulfilling)
    #
    # defeating = PrettyTable()
    # defeating.field_names = ["Run", "Feature number", "False Negative", "True Positive",
    #                      "True Negative", "False positive", "p-value"]
    # for i in range(100):
    #     stream = influential_stream.InfluentialStream(self_defeating=0.999, self_fulfilling=1)
    #     evaluating(stream, i, defeating)

    print("equal")
    print(equal)
    # print("defeating")
    # print(defeating)
    # print("fulfilling")
    # print(fulfilling)


def evaluating(stream, run, x):
    classifier = naive_bayes.NaiveBayes()
    # classifier = PerceptronMask()
    # classifier = HoeffdingTreeClassifier()
    # classifier = PassiveAggressiveClassifier()

    # 3. Setup the evaluator
    evaluator = evaluate_influential.EvaluateInfluential(show_plot=False,
                                                         pretrain_size=200,
                                                         max_samples=2200,
                                                         batch_size=1,
                                                         n_time_windows=2,
                                                         n_intervals=8,
                                                         metrics=['accuracy'],
                                                         data_points_for_classification=False,
                                                         weight_output=True)

    pipe = Pipeline([('Naive Bayes', classifier)])

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=pipe)
    for result in evaluator.table_result:
        result.insert(0, run)
        x.add_row(result)


if __name__ == '__main__':
    demo()
