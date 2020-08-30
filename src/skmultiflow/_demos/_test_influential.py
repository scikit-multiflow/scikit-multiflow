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
    runs = 100

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
    x_accuracy = [[] for _ in range(3)]
    final_weights = [[] for _ in range(3)]

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
    weightA = [1, 1.002, 0.995]
    weightB = [1, 0.998, 1.005]
    for i in range(runs):
        component_pos_drift = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state=50,
                                                      n_classes=2, n_features=1,
                                                      n_centroids=1, num_drift_centroids=1,
                                                      change_speed=(1 / runs) * i,
                                                      class_weights=[0, 1])
        component_pos = RandomRBFGenerator(model_random_state=99, sample_random_state=50,
                                           n_classes=2, n_features=1, n_centroids=1, class_weights=[0, 1])
        component_neg = RandomRBFGenerator(model_random_state=99, sample_random_state=88,
                                           n_classes=2, n_features=1, n_centroids=1, class_weights=[1, 0])
        component_neg_drift = RandomRBFGeneratorDrift(model_random_state=101, sample_random_state=88,
                                                      n_classes=2, n_features=1,
                                                      n_centroids=1, num_drift_centroids=1,
                                                      change_speed=(1 / runs) * i,
                                                      class_weights=[1, 0])

        for j in range(3):
            stream = influential_stream.InfluentialStream(self_fulfilling=weightA[j], self_defeating=weightB[j],
                                                          streams=[component_pos_drift] * 5 + [component_neg_drift] * 5)
            evaluating(stream, i, positive_table[j], negative_table[j], influence_on_positive[j],
                       influence_on_negative[j],
                       abs_mean_pos[j], abs_mean_neg[j], accuracy[j], x_accuracy[j], final_weights[j])

    for i in range(3):
        print(positive_table[i])
        print(negative_table[i])
        print("how many times a p value below 0.01:", len(x_accuracy[i]))

    y = []
    i = 0
    for item in influence_on_positive[0]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_positive[0] = list(filter(lambda a: a is not None, influence_on_positive[0]))

    plt.figure(1)
    plt.plot(y, influence_on_positive[0], label="Influence on positive instances", color="green")
    y = []
    i = 0
    for item in influence_on_negative[0]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_negative[0] = list(filter(lambda a: a is not None, influence_on_negative[0]))
    plt.plot(y, influence_on_negative[0], label="Influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p value')
    plt.title('Equal weights')
    plt.legend()

    y = list(range(0, runs, 1))
    plt.figure(2)
    plt.plot(y, abs_mean_pos[0], label="Absolute difference in mean positive instances", color="green")
    plt.plot(y, abs_mean_neg[0], label="Absolute difference in mean negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('Absolute difference in mean')
    plt.title('Absolute difference in mean in equal approach')
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
    plt.title('Self fulfilling approach')
    plt.legend()

    y = list(range(0, runs, 1))
    plt.figure(4)
    plt.plot(y, abs_mean_pos[1], label="Absolute difference in mean positive instances", color="green")
    plt.plot(y, abs_mean_neg[1], label="Absolute difference in mean negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('Absolute difference in mean')
    plt.title('Absolute difference in mean in Self fulfilling approach')
    plt.legend()

    y = []
    i = 0
    plt.figure(5)
    for item in influence_on_positive[2]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_positive[2] = list(filter(lambda a: a is not None, influence_on_positive[2]))
    plt.plot(y, influence_on_positive[2], label="Influence on positive instances", color="green")
    y = []
    i = 0
    for item in influence_on_negative[2]:
        if item is not None:
            y.append(i)
        i += 1
    influence_on_negative[2] = list(filter(lambda a: a is not None, influence_on_negative[2]))
    plt.plot(y, influence_on_negative[2], label="Influence on negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('p-value')
    plt.title('Self defeating approach')
    plt.legend()

    y = list(range(0, runs, 1))
    plt.figure(6)
    plt.plot(y, abs_mean_pos[2], label="absolute difference in mean positive instances", color="green")
    plt.plot(y, abs_mean_neg[2], label="absolute difference in mean negative instances", color="red")
    plt.xlabel('runs')
    plt.ylabel('Absolute difference in mean')
    plt.title('Absolute difference in mean in Self defeating approach')
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
        accuracy[i] = [accuracy[i][j] for j in x_accuracy[i]]
    plt.plot(x_accuracy[0], accuracy[0], label="accuracy without influence", color="grey")
    plt.plot(x_accuracy[1], accuracy[1], label="accuracy in self fulfilling approach", color="blue")
    plt.plot(x_accuracy[2], accuracy[2], label="accuracy in self defeating approach", color="purple")
    plt.xlabel('runs')
    plt.ylabel("accuracy")
    plt.title("Accuracy for strategies with at least one p value <0.05")
    plt.legend()

    title = ['equal weights', 'self fulfilling approach', 'self defeating approach']
    figure = 9
    for j in range(2):
        plt.figure(figure)
        x_value = list(range(0, len(final_weights[j + 1]), 1))
        for i in range(len(stream.streams)):
            label = "stream {}".format(i)
            y_values = [weight[i] for weight in final_weights[j + 1]]
            plt.plot(x_value, y_values, label=label)
        plt.xlabel('runs')
        plt.ylabel("weight")
        plt.title("Final weights of streams with  {}".format(title[j + 1]))
        plt.legend()
        figure += 1

    pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
    for fig in range(1, plt.figure().number):
        pdf.savefig(fig)
    pdf.close()
    plt.show()


def evaluating(stream, run, pos, neg, influence_pos, influence_neg, abs_mean_pos, abs_mean_neg, accuracy, x_accuracy,
               final_weights):
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
                                                         weight_output=True,
                                                         weight_plot=False)
    pipe = Pipeline([('Naive Bayes', classifier)])

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=pipe)
    final_weights.append(evaluator.stream.weight)
    accuracy.append(evaluator.accuracy[0])
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
                if result[7] < 0.01:
                    idx_neg.append(run)
            influence_neg.append(result[7])
            abs_mean_neg.append(result[6])
            neg.add_row(result)
    # idx = [i for i, j in zip(idx_pos, idx_neg) if i == j]
    idx = idx_pos + list(set(idx_neg) - set(idx_pos))
    x_accuracy.extend(idx)


if __name__ == '__main__':
    demo()
