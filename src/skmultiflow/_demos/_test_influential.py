from skmultiflow.data.influential_stream import InfluentialStream
from skmultiflow.evaluation.evaluate_influential import EvaluateInfluential
from skmultiflow.bayes.naive_bayes import NaiveBayes
from skmultiflow.core import Pipeline
from prettytable import PrettyTable
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
from skmultiflow.drift_detection.prediction_influence_detector import predictionInfluenceDetector
import matplotlib.pyplot as plt


def demo():
    """ _test_influential

    This demo tests if the streams are correctly created and
    if the classifier chooses a new sample based on the weights
    of the streams.

    :return:
    """
    runs = 100
    n_comparisons = 1
    n_features = 1
    info_positive, info_negative = [], []
    accuracy = [[] for _ in range(3)]
    final_weights = [[] for _ in range(3)]
    weightA = [1, 1.003, 0.997]
    weightB = [1, 0.997, 1.003]
    for i in range(runs):
        list_of_streams = []
        for x in range(5):
            list_of_streams.append(RandomRBFGeneratorDrift(model_random_state=101, sample_random_state=51,
                                                           n_classes=2, n_features=n_features,
                                                           n_centroids=1, num_drift_centroids=1,
                                                           change_speed=(1 / runs) * i,
                                                           class_weights=[1, 0]))
        for x in range(5):
            list_of_streams.append(RandomRBFGeneratorDrift(model_random_state=101, sample_random_state=50,
                                                           n_classes=2, n_features=n_features,
                                                           n_centroids=1, num_drift_centroids=1,
                                                           change_speed=(1 / runs) * i,
                                                           class_weights=[0, 1]))
        for strategy in range(len(weightA)):
            stream = InfluentialStream(self_fulfilling=weightA[strategy], self_defeating=weightB[strategy],
                                       streams=list_of_streams)
            classifier = NaiveBayes()

            # Setup the evaluator
            evaluator = EvaluateInfluential(show_plot=False,
                                            pretrain_size=200,
                                            max_samples=2200,
                                            batch_size=1,
                                            batch_size_update=False,
                                            n_time_windows=n_comparisons + 1,
                                            n_intervals=8,
                                            metrics=['accuracy'],
                                            data_points_for_classification=False,
                                            weight_output=False,
                                            weight_plot=False)
            pipe = Pipeline([('Naive Bayes', classifier)])

            # 4. Run evaluation
            evaluator.evaluate(stream=stream, model=pipe)

            # 5. detect prediction influence
            detection = predictionInfluenceDetector(run=i, strategy=strategy)
            detection.calculate_density(evaluator)
            info_positive.extend(detection.table_influence_on_positive)
            info_negative.extend(detection.table_influence_on_negative)

            # add results of the run to certain lists
            final_weights[strategy].append(evaluator.stream.weight)
            accuracy[strategy].append(evaluator.accuracy[0])

    # divide big information lists in sub lists
    run, strategy, feature, chunk, p_value, absolute_difference, mean_true, mean_false = 0, 1, 2, 3, 9, 8, 5, 7
    no_influence_pos = [item for item in info_positive if item[strategy] == 0]
    self_fulfilling_pos = [item for item in info_positive if item[strategy] == 1]
    self_defeating_pos = [item for item in info_positive if item[strategy] == 2]
    no_influence_neg = [item for item in info_negative if item[strategy] == 0]
    self_fulfilling_neg = [item for item in info_negative if item[strategy] == 1]
    self_defeating_neg = [item for item in info_negative if item[strategy] == 2]

    # visualize results

    title = ['no prediction influence', 'self fulfilling approach', 'self defeating approach']
    positive_instances = [no_influence_pos, self_fulfilling_pos, self_defeating_pos]
    negative_instances = [no_influence_neg, self_fulfilling_neg, self_defeating_neg]
    plt.figure(1)
    for i in range(len(positive_instances)):
        for j in range(n_comparisons):
            for f in range(n_features):
                plt.plot([item[0] for item in positive_instances[i] if
                          item[p_value] < 0.01 and item[chunk] == j and item[feature] == f],
                         [item[p_value] for item in positive_instances[i] if
                          item[p_value] < 0.01 and item[chunk] == j and item[feature] == f], label=title[i])
    plt.xlabel('Runs')
    plt.ylabel('Average p-value')
    plt.title('Average p-value per strategy per run on positive instances')
    plt.legend()

    plt.figure(2)
    for i in range(len(negative_instances)):
        for j in range(n_comparisons):
            for f in range(n_features):
                plt.plot([item[0] for item in negative_instances[i] if
                          item[p_value] < 0.01 and item[chunk] == j and item[feature] == f],
                         [item[p_value] for item in negative_instances[i] if
                          item[p_value] < 0.01 and item[chunk] == j and item[feature] == f], label=title[i])
    plt.xlabel('Runs')
    plt.ylabel('Average p-value')
    plt.title('Average p-value per strategy per run on negative instances')
    plt.legend()

    plt.figure(3)
    for i in range(len(positive_instances)):
        for j in range(n_comparisons):
            for f in range(n_features):
                plt.plot([item[0] for item in positive_instances[i] if item[chunk] == j and item[feature] == f],
                         [item[mean_true] for item in positive_instances[i] if item[chunk] == j and item[feature] == f],
                         label="True positive " + title[i])
                plt.plot([item[0] for item in positive_instances[i] if item[chunk] == j and item[feature] == f],
                         [item[mean_false] for item in positive_instances[i] if
                          item[chunk] == j and item[feature] == f],
                         label="False negative " + title[i])
    plt.xlabel('Runs')
    plt.ylabel('mean sample value')
    plt.title('Mean sample value in positive instances')
    plt.legend()

    plt.figure(4)
    for i in range(len(negative_instances)):
        for j in range(n_comparisons):
            for f in range(n_features):
                plt.plot([item[0] for item in negative_instances[i] if item[chunk] == j and item[feature] == f],
                         [item[mean_true] for item in negative_instances[i] if item[chunk] == j and item[feature] == f],
                         label="True negative " + title[i])
                plt.plot([item[0] for item in negative_instances[i] if item[chunk] == j and item[feature] == f],
                         [item[mean_false] for item in negative_instances[i] if
                          item[chunk] == j and item[feature] == f],
                         label="False positive " + title[i])
    plt.xlabel('Runs')
    plt.ylabel('mean sample value')
    plt.title('Mean sample value in negative instances')
    plt.legend()

    plt.figure(5)
    for i in range(len(negative_instances)):
        for j in range(n_comparisons):
            for f in range(n_features):
                plt.plot([item[0] for item in negative_instances[i] if item[chunk] == j and item[feature] == f],
                         [item[absolute_difference] for item in negative_instances[i] if
                          item[chunk] == j and item[feature] == f],
                         label="Absolute difference mean " + title[i])
    plt.xlabel('Runs')
    plt.ylabel('y')
    plt.title('Absolute difference in negative instances')
    plt.legend()

    plt.figure(6)
    for i in range(len(positive_instances)):
        for j in range(n_comparisons):
            for f in range(n_features):
                plt.plot([item[0] for item in positive_instances[i] if item[chunk] == j and item[feature] == f],
                         [item[absolute_difference] for item in positive_instances[i] if
                          item[chunk] == j and item[feature] == f],
                         label="Absolute difference mean " + title[i])
    plt.xlabel('Runs')
    plt.ylabel('y')
    plt.title('Absolute difference in positive instances')
    plt.legend()

    plt.figure(7)
    for i in range(len(title)):
        plt.plot(list(range(0, runs, 1)), accuracy[i], label=title[i])
    plt.xlabel('Runs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy classifier per run per strategy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo()
