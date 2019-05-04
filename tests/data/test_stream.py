import pandas as pd
import time
import streamz
from skmultiflow.data import from_generator, DataGenerator

def test_sink_to_list():
    df = pd.DataFrame(pd.np.random.randint(0, 100,size=(300, 4)), columns=list('ABCD'))
    gen = DataGenerator(df)
    stream = streamz.Source.from_generator(gen, batch_size=20, poll_interval=.001)

    sub_df_list = stream.sink_to_list()
    stream.start()
    time.sleep(1.0)  # sleep long enough to let all sub_df pass through the stream
    pull_from_stream_df = pd.concat(sub_df_list)

    pd.testing.assert_frame_equal(pull_from_stream_df, df)
