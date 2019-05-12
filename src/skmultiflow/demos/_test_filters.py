import os
import time
import pandas as pd
from streamz import Source

from src.skmultiflow.data import DataGenerator
from sklearn.impute import SimpleImputer

curr_dir = dir_path = os.path.dirname(os.path.realpath(__file__))

def demo():
    """ _test_filters

    This demo test compatibility between `sklearn.impute`. The transform is set
    to clean any value equal to -47, replacing it with the median value
    of the last 10 samples, or less if there aren't 10 samples available.

    The output will be the 10 instances used in the transform. The first
    9 are kept untouched, as they don't have any feature value of -47. The
    last samples has its first feature value equal to -47, so it's replaced
    by the median of the 9 first samples.

    """
    df = pd.read_csv(os.path.join(curr_dir, '../data/datasets/covtype.csv'))
    gen = DataGenerator(df, return_np=True)
    stream = Source.from_generator(gen, batch_size=1, poll_interval=.001)
    sliding_array = stream.sliding_window(10).map(pd.np.concatenate)  # get last ten elements
    data_from_stream = sliding_array.map(SimpleImputer(missing_values=-47, strategy='median').fit_transform).sink_to_list()
    stream.start()
    time.sleep(1)  # wait long enough to let all batches pass through the stream
    df_from_stream = pd.DataFrame(pd.np.concatenate(data_from_stream))

    print(f"number of -47 values before imputation {(df == -47).sum().sum()}")
    print(f"number of -47 values after imputation {(df_from_stream == -47).sum().sum()}")



if __name__ == '__main__':
    demo()
