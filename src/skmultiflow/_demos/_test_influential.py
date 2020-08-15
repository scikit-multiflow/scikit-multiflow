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
import matplotlib.backends.backend_pdf


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

    abs_mean_pos = [[] for _ in range(3)]
    abs_mean_neg = [[] for _ in range(3)]
    influence_on_positive = [[] for _ in range(3)]
    influence_on_negative = [[] for _ in range(3)]
    accuracy = [[] for _ in range(3)]

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
    runs = 100

    for i in range(runs):
        component_pos = RandomRBFGeneratorDrift(n_classes=2, n_features=3, n_centroids=1, change_speed=(1 / runs) * i,
                                                num_drift_centroids=1, class_weights=[0.25, 0.75])
        component_neg = RandomRBFGeneratorDrift(n_classes=2, n_features=3, n_centroids=1, change_speed=(1 / runs) * i,
                                                num_drift_centroids=1, class_weights=[0.75, 0.25])
        component = RandomRBFGeneratorDrift(n_classes=2, n_features=3, n_centroids=1, change_speed=(1 / runs) * i,
                                            num_drift_centroids=1, class_weights=[0.5, 0.5])

        stream = influential_stream.InfluentialStream(self_defeating=1, self_fulfilling=1,
                                                      streams=[component_pos]*3 + [component_neg]*3 + [component]*3)
        evaluating(stream, i, equal_pos, equal_neg, influence_on_positive[0], influence_on_negative[0],
                   abs_mean_pos[0], abs_mean_neg[0], accuracy[0])

        stream = influential_stream.InfluentialStream(self_defeating=0.999, self_fulfilling=1.001,
                                                      streams=[component_pos]*3 + [component_neg]*3 + [component]*3)
        evaluating(stream, i, fulfilling_pos, fulfilling_neg, influence_on_positive[1], influence_on_negative[1],
                   abs_mean_pos[1], abs_mean_neg[1], accuracy[1])

        stream = influential_stream.InfluentialStream(self_defeating=1.001, self_fulfilling=0.999,
                                                      streams=[component_pos]*3 + [component_neg]*3 + [component]*3)

        evaluating(stream, i, defeating_pos, defeating_neg, influence_on_positive[2], influence_on_negative[2],
                   abs_mean_pos[2], abs_mean_neg[2], accuracy[2])

    print(fulfilling_pos)
    print(fulfilling_neg)

    y = []
    i = 0
    for item in influence_on_positive[0]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_positive[0] = list(filter(lambda a: a is not None, influence_on_positive[0]))

    plt.figure(1)
    plt.plot(y, influence_on_positive[0], label="influence on positive instances", color="green")
    y = []
    i = 0
    for item in influence_on_negative[0]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_negative[0] = list(filter(lambda a: a is not None, influence_on_negative[0]))
    plt.plot(y, influence_on_negative[0], label="influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p value')
    plt.title('Equal weights')
    plt.legend()

    y = list(range(0, runs, 1))
    plt.figure(2)
    plt.plot(y, abs_mean_pos[0], label="absolute difference of mean positive instances", color="green")
    plt.plot(y, abs_mean_neg[0], label="absolute difference of mean negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('Absolute difference in mean')
    plt.title('Abs difference in mean in equal approach')
    plt.legend()

    y = []
    i = 0
    for item in influence_on_positive[1]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_positive[1] = list(filter(lambda a: a is not None, influence_on_positive[1]))
    plt.figure(3)
    plt.plot(y, influence_on_positive[1], label="influence on positive instances", color="green")
    y = []
    i = 0
    for item in influence_on_negative[1]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_negative[1] = list(filter(lambda a: a is not None, influence_on_negative[1]))
    plt.plot(y, influence_on_negative[1], label="influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p value')
    plt.title('Self fulfilling')
    plt.legend()

    y = list(range(0, runs, 1))
    plt.figure(4)
    plt.plot(y, abs_mean_pos[1], label="absolute difference of mean positive instances", color="green")
    plt.plot(y, abs_mean_neg[1], label="absolute difference of mean negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('Absolute difference in mean')
    plt.title('Abs difference in mean in Self fulfilling approach')
    plt.legend()

    y = []
    i = 0
    plt.figure(5)
    for item in influence_on_positive[2]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_positive[2] = list(filter(lambda a: a is not None, influence_on_positive[2]))
    plt.plot(y, influence_on_positive[2], label="influence on positive instances", color="green")
    y = []
    i = 0
    for item in influence_on_negative[2]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_negative[2] = list(filter(lambda a: a is not None, influence_on_negative[2]))
    plt.plot(y, influence_on_negative[2], label="influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p value')
    plt.title('Self defeating')
    plt.legend()

    y = list(range(0, runs, 1))
    plt.figure(6)
    plt.plot(y, abs_mean_pos[2], label="absolute difference of mean positive instances", color="green")
    plt.plot(y, abs_mean_neg[2], label="absolute difference of mean negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('Absolute difference in mean')
    plt.title('Abs difference in mean in Self defeating approach')
    plt.legend()

    plt.figure(7)
    plt.plot(y, accuracy[0], label="accuracy without influence", color="grey")
    plt.plot(y, accuracy[1], label="accuracy in self fulfilling approach", color="blue")
    plt.plot(y, accuracy[2], label="accuracy in self defeating approach", color="purple")
    plt.xlabel('runs')
    plt.ylabel("accuracy")
    plt.title("accuracy self defeating approach")
    plt.legend()
    pdf = matplotlib.backends.backend_pdf.PdfPages("testtesttest.pdf")
    for fig in range(1, plt.figure().number):
        pdf.savefig(fig)
    pdf.close()
    plt.show()


def evaluating(stream, run, pos, neg, influence_pos, influence_neg, abs_mean_pos, abs_mean_neg, accuracy):
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
                                                         n_intervals=6,
                                                         metrics=['accuracy'],
                                                         data_points_for_classification=False,
                                                         weight_output=True)
    pipe = Pipeline([('Naive Bayes', classifier)])

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=pipe)

    accuracy.append(evaluator.accuracy)

    for result in evaluator.table_influence_on_positive:
        result.insert(0, run)
        if result[1] == 0:
            influence_pos.append(result[7])
            abs_mean_pos.append(result[6])
            pos.add_row(result)

    for result in evaluator.table_influence_on_negative:
        result.insert(0, run)
        if result[1] == 0:
            influence_neg.append(result[7])
            abs_mean_neg.append(result[6])
            neg.add_row(result)


if __name__ == '__main__':
    demo()
