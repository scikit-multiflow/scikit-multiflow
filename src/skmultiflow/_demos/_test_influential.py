import operator

from skmultiflow.data import influential_stream, random_rbf_generator
from skmultiflow.evaluation import evaluate_influential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import naive_bayes
from skmultiflow.core import Pipeline
from prettytable import PrettyTable


def demo():
    """ _test_influential

    This demo tests if the streams are correctly created and
    if the classifier chooses a new sample based on the weights
    of the streams.

    :return:
    """
    equal = PrettyTable()
    equal.field_names = ["Run", "Feature number", "pos/neg influence", "FN/TN sample", "FNTP sample", "pvalue"]
    for i in range(3):
        stream = influential_stream.InfluentialStream(self_defeating=0.999, self_fulfilling=1.1)
        evaluating(stream, i, equal)

    print(equal)

    # fulfilling = PrettyTable()
    # fulfilling.field_names = ["Run", "Feature number", "pos/neg influence", "pvalue"]
    # for i in range(10):
    #     stream = influential_stream.InfluentialStream(self_defeating=0.5, self_fulfilling=2)
    #     evaluating(stream, i, fulfilling)
    #
    # print(fulfilling)
    #
    # defeating = PrettyTable()
    # defeating.field_names = ["Run", "Feature number", "pos/neg influence", "pvalue"]
    # for i in range(10):
    #     stream = influential_stream.InfluentialStream(self_defeating=1.5, self_fulfilling=0.5)
    #     evaluating(stream, i, defeating)
    # print(defeating)


def evaluating(stream, run, x):
    classifier = naive_bayes.NaiveBayes()
    # classifier = PerceptronMask()
    # classifier = HoeffdingTreeClassifier()
    # classifier = PassiveAggressiveClassifier()

    # 3. Setup the evaluator
    evaluator = evaluate_influential.EvaluateInfluential(show_plot=False,
                                                         pretrain_size=200,
                                                         max_samples=2000,
                                                         batch_size=1,
                                                         n_time_windows=2,
                                                         n_intervals=4,
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
