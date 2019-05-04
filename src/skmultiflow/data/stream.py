import pandas as pd
from tornado import gen
from streamz import Stream, Source
from skmultiflow.data.base_generator import BaseGenerator

@Stream.register_api(staticmethod)
class from_generator(Source):
    def __init__(self, mf_gen, batch_size, poll_interval=1.0, start=False):
        assert isinstance(mf_gen, BaseGenerator)
        super(from_generator, self).__init__(ensure_io_loop=True)
        self.batch_size = batch_size
        self.stopped = True
        self.started = False
        self.mf_gen = mf_gen
        self.poll_interval = poll_interval
        if start:
            self.start()
    def start(self):
        self.stopped = False
        self.started = True
        self.loop.add_callback(self.do_poll)

    @gen.coroutine
    def do_poll(self):
        while self.mf_gen.has_more_samples():
            sample = self.mf_gen.next_sample(self.batch_size)
            yield self._emit(sample)
            yield gen.sleep(self.poll_interval)
