from skmultiflow.data.source.data_source import DataSource
import threading


class GeneratorDataSource(DataSource):
    """ GeneratorDataSource class.
    Provides a DataSource implementation, pulling data from a generator.
    """

    def __init__(self, record_to_dictionary, observers, generator):
        super().__init__(record_to_dictionary, observers)
        self.generator = generator

        self.name = "GeneratorDataSource: {}".format(self.generator.name)


    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        pass

    def listen_for_events(self):
        thread = threading.Thread(target=self.consume_generator_messages, args=())
        thread.daemon = True
        thread.start()

    def consume_generator_messages(self):
        event = self.generator.next_sample()
        while event is not None:
            self.on_new_event(event)
            event = self.generator.next_sample()
