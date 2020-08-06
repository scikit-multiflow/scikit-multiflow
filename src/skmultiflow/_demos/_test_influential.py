import operator

from skmultiflow.data import influential_stream, random_rbf_generator
from skmultiflow.evaluation import evaluate_influential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import naive_bayes
from skmultiflow.core import Pipeline
from prettytable import PrettyTable
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
import matplotlib.pyplot as plt


def demo():
    """ _test_influential

    This demo tests if the streams are correctly created and
    if the classifier chooses a new sample based on the weights
    of the streams.

    :return:
    """
    equal_pos = PrettyTable()
    equal_neg = PrettyTable()
    fulfilling_pos = PrettyTable()
    fulfilling_neg = PrettyTable()
    defeating_pos = PrettyTable()
    defeating_neg = PrettyTable()
    table_names_pos = ["Run", "Feature number", "length subset TP", "TP sample mean", "length subset FN",
                       "FN sample mean", "abs difference in mean", "p value"]
    table_names_neg = ["Run", "Feature number", "length subset TN", "TN sample mean", "length subset FP",
                       "FP sample mean", "abs difference in mean", "p value"]

    defeating_pos.field_names = table_names_pos
    equal_pos.field_names = table_names_pos
    fulfilling_pos.field_names = table_names_pos
    defeating_neg.field_names = table_names_neg
    equal_neg.field_names = table_names_neg
    fulfilling_neg.field_names = table_names_neg

    positive_influence_fulfilling = []
    negative_influence_fulfilling = []

    positive_influence_equal = []
    negative_influence_equal = []

    positive_influence_defeating = []
    negative_influence_defeating = []

    runs = 100

    for i in range(runs):
        stream = influential_stream.InfluentialStream(self_defeating=1, self_fulfilling=1,
                                                      streams=[RandomRBFGenerator(model_random_state=99,
                                                                                  sample_random_state=50,
                                                                                  n_classes=2, n_features=4,
                                                                                  n_centroids=20),
                                                               RandomRBFGeneratorDrift(model_random_state=99,
                                                                                       sample_random_state=50,
                                                                                       n_classes=2,
                                                                                       n_features=4,
                                                                                       n_centroids=20,
                                                                                       change_speed=0.8,
                                                                                       num_drift_centroids=int((20/runs)*i))])
        evaluating(stream, i, equal_pos, equal_neg, positive_influence_equal, negative_influence_equal)

    for i in range(runs):
        stream = influential_stream.InfluentialStream(self_defeating=1, self_fulfilling=1.001,
                                                      streams=[RandomRBFGenerator(model_random_state=99,
                                                                                  sample_random_state=50,
                                                                                  n_classes=2, n_features=4,
                                                                                  n_centroids=20),
                                                               RandomRBFGeneratorDrift(model_random_state=99,
                                                                                       sample_random_state=50,
                                                                                       n_classes=2,
                                                                                       n_features=4,
                                                                                       n_centroids=20,
                                                                                       change_speed=0.8,
                                                                                       num_drift_centroids=int(
                                                                                           (20 / runs) * i))])
        evaluating(stream, i, fulfilling_pos, fulfilling_neg, positive_influence_fulfilling,
                   negative_influence_fulfilling)

    for i in range(runs):
        stream = influential_stream.InfluentialStream(self_defeating=0.999, self_fulfilling=1,
                                                      streams=[RandomRBFGenerator(model_random_state=99,
                                                                                  sample_random_state=50,
                                                                                  n_classes=2, n_features=4,
                                                                                  n_centroids=20),
                                                               RandomRBFGeneratorDrift(model_random_state=99,
                                                                                       sample_random_state=50,
                                                                                       n_classes=2,
                                                                                       n_features=4,
                                                                                       n_centroids=20,
                                                                                       change_speed=0.8,
                                                                                       num_drift_centroids=int((20/runs)*i))])
        evaluating(stream, i, defeating_pos, defeating_neg, positive_influence_defeating, negative_influence_defeating)

    # print("equal")
    print(equal_pos)
    print(equal_neg, fulfilling_pos, fulfilling_neg, defeating_pos, defeating_neg)
    y = [*range(0, runs, 1)]
    plt.figure(1)
    plt.plot(y, positive_influence_equal, label="influence on positive instances", color="green")
    plt.plot(y, negative_influence_equal, label="influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p value')
    plt.title('Equal weights')

    plt.figure(2)
    plt.plot(y, positive_influence_fulfilling, label="influence on positive instances", color="green")
    plt.plot(y, negative_influence_fulfilling, label="influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p value')
    plt.title('Self fulfilling')
    plt.legend()

    plt.figure(3)
    plt.plot(y, positive_influence_defeating, label="influence on positive instances", color="green")
    plt.plot(y, negative_influence_defeating, label="influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p value')
    plt.title('Self defeating')
    plt.legend()

    plt.show()

    # print("defeating")
    # print(defeating)
    # print("fulfilling")
    # print(fulfilling)


def evaluating(stream, run, pos, neg, positive_influence, negative_influence):
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
                                                         n_intervals=4,
                                                         metrics=['accuracy'],
                                                         data_points_for_classification=False,
                                                         weight_output=True)

    pipe = Pipeline([('Naive Bayes', classifier)])

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=pipe)
    for result in evaluator.table_positive_influence:
        result.insert(0, run)
        if result[1] == 0:
            positive_influence.append(result[6])
            pos.add_row(result)

    for result in evaluator.table_negative_influence:
        result.insert(0, run)
        if result[1] == 0:
            neg.add_row(result)
            negative_influence.append(result[6])


if __name__ == '__main__':
    demo()
