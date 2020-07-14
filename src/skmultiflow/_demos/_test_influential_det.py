from skmultiflow.data import influential_stream
from skmultiflow.evaluation import evaluate_influential
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import file_stream


def demo():
    filedStream = file_stream.FileStream("/Users/tinekejelsma/Downloads/agr_a.csv")
    filedStream.name = "Agrawal - dataset"
    stream = influential_stream.InfluentialStream(streams=[filedStream], self_defeating=1, self_fulfilling=1)

    ht = HoeffdingTreeClassifier()

    # 3. Setup the evaluator
    evaluator = evaluate_influential.EvaluateInfluential(show_plot=True,
                                                         pretrain_size=200,
                                                         max_samples=20000,
                                                         batch_size=1,
                                                         n_time_windows=2,
                                                         n_intervals=4,
                                                         metrics=['accuracy', 'kappa', 'data_points'],
                                                         data_points_for_classification=False)

    print("weights: ", stream.weight)

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=ht)


if __name__ == '__main__':
    demo()
