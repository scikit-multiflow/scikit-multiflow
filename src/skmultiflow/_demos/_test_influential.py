from skmultiflow.data import influential_stream, random_rbf_generator
from skmultiflow.evaluation import evaluate_influential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import naive_bayes
from skmultiflow.core import Pipeline
from prettytable import PrettyTable
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


def demo():
    """ _test_influential

    This demo tests if the streams are correctly created and
    if the classifier chooses a new sample based on the weights
    of the streams.

    :return:
    """
    runs = 50

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
    y_accuracy = [[] for _ in range(3)]

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

    negative_table = [equal_neg, fulfilling_neg, defeating_neg]
    positive_table = [equal_pos, fulfilling_pos, defeating_pos]
    weightA = [1, 1.001, 0.999]
    weightB = [1, 0.999, 1.001]
    for i in range(runs):
        component_pos_drift = RandomRBFGeneratorDrift(n_classes=2, n_features=1, n_centroids=1, num_drift_centroids=1,
                                                      change_speed=0, class_weights=[0, 1])

        # component_pos = RandomRBFGenerator(n_classes=2, n_features=2, n_centroids=2, class_weights=[0.4, 0.6])
        # component_neg = RandomRBFGenerator(n_classes=2, n_features=2, n_centroids=20, class_weights=[0.6, 0.4])
        # component_neg_drift = ConceptDriftStream(stream=RandomRBFGenerator(n_classes=2, n_features=2,
        #                                                                    n_centroids=20, class_weights=[0.7, 0.3]),
        #                                          drift_stream=RandomRBFGeneratorDrift(n_classes=2, n_features=2,
        #                                                                               n_centroids=20,
        #                                                                               num_drift_centroids=int(
        #                                                                                   (20 / runs) * i),
        #                                                                               change_speed=(1 / runs) * i,
        #                                                                               class_weights=[0.5, 0.5]),
        #                                          position=0, width=1000)
        component_neg_drift = RandomRBFGeneratorDrift(n_classes=2, n_features=1, n_centroids=1, num_drift_centroids=1,
                                                      change_speed=1, class_weights=[1, 0])

        for j in range(3):
            stream = influential_stream.InfluentialStream(self_defeating=weightA[j], self_fulfilling=weightB[j],
                                                          streams=[component_neg_drift] * 5 + [component_pos_drift] * 5)
            evaluating(stream, i, positive_table[j], negative_table[j], influence_on_positive[j],
                       influence_on_negative[j],
                       abs_mean_pos[j], abs_mean_neg[j], accuracy[j], y_accuracy[j])

    print(positive_table[0])
    print(negative_table[0])
    print(positive_table[1])
    print(negative_table[1])
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
    plt.title("accuracy per strategy")
    plt.legend()

    plt.figure(8)
    for i in range(3):
        accuracy[i] = [accuracy[i][j] for j in y_accuracy[i]]
    plt.plot(y_accuracy[0], accuracy[0], label="accuracy without influence", color="grey")
    plt.plot(y_accuracy[1], accuracy[1], label="accuracy in self fulfilling approach", color="blue")
    plt.plot(y_accuracy[2], accuracy[2], label="accuracy in self defeating approach", color="purple")
    plt.xlabel('runs')
    plt.ylabel("accuracy")
    plt.title("accuracy for runs with p value <0.05")
    plt.legend()

    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
    for fig in range(1, plt.figure().number):
        pdf.savefig(fig)
    pdf.close()
    plt.show()


def evaluating(stream, run, pos, neg, influence_pos, influence_neg, abs_mean_pos, abs_mean_neg, accuracy, y_accuracy):
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
    accuracy.append(evaluator.accuracy)
    idx_pos = []
    idx_neg = []
    for result in evaluator.table_influence_on_positive:
        result.insert(0, run)
        if result[1] == 0:
            if result[7] is not None:
                if result[7] < 0.05:
                    idx_pos.append(run)
            influence_pos.append(result[7])
            abs_mean_pos.append(result[6])
            pos.add_row(result)

    for result in evaluator.table_influence_on_negative:
        result.insert(0, run)
        if result[1] == 0:
            if result[7] is not None:
                if result[7] < 0.05:
                    idx_neg.append(run)
            influence_neg.append(result[7])
            abs_mean_neg.append(result[6])
            neg.add_row(result)
    # idx = [i for i, j in zip(idx_pos, idx_neg) if i == j]
    idx = idx_pos + list(set(idx_neg) - set(idx_pos))
    y_accuracy.extend(idx)


if __name__ == '__main__':
    demo()
