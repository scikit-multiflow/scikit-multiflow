import pandas as pd
import time
import streamz
import streamz.utils_test
from skmultiflow.data import from_multiflow_generator, DataGenerator

def test_sink_to_list():
    """
    1. send data from a generator into a streaming dataframe
    2. concatenate what is emitted from the stream into a pandas dataframe
    3. await until the concatenated result is equal to the dataframe of origin
    """
    df = pd.DataFrame(pd.np.random.randint(0, 100,size=(300, 4)), columns=list('ABCD'))
    gen = DataGenerator(df)
    stream = streamz.Source.from_multiflow_generator(gen, batch_size=20, poll_interval=.001)

    result_list = [pd.DataFrame(columns=df.columns)]
    stream.sink(result_list.append)
    stream.start()

    predicate = lambda : pd.np.array_equal(pd.concat(result_list, axis=0).values, df.values)
    streamz.utils_test.await_for(predicate, 0.2)



def test_to_dataframe():
    """
    1. send data from a generator into a streaming dataframe
    2. concatenate what is emitted from the streaming dataframe into a pandas dataframe
    3. await until the concatenated result is equal to the dataframe of origin
    """
    df = pd.DataFrame(pd.np.random.randint(0, 100,size=(300, 4)), columns=list('ABCD'))
    gen = DataGenerator(df)
    stream = streamz.Source.from_multiflow_generator(gen, batch_size=20, poll_interval=.01)
    sdf = stream.to_dataframe(example=df.head(1))

    result_list = [pd.DataFrame(columns=df.columns)]
    sdf.stream.sink(result_list.append)
    stream.start()

    predicate = lambda : pd.np.array_equal(pd.concat(result_list, axis=0).values, df.values)
    streamz.utils_test.await_for(predicate, 0.2)

def test_destroy():
    df = pd.DataFrame(pd.np.random.randint(0, 100,size=(300, 4)), columns=list('ABCD'))
    gen = DataGenerator(df)
    stream = streamz.Source.from_multiflow_generator(gen, batch_size=20, poll_interval=.01)
    new_node = stream.map(lambda df : df['A'].min())

    stream.start()
    new_node.destroy()
    assert not stream.downstreams
