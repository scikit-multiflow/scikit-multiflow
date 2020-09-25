from skmultiflow.data.source import DataSource


class GeneratorDataSource(DataSource):
    """ GeneratorDataSource class.
    Provides a DataSource implementation, pulling data from a generator.
    """

    def __init__(self, record_to_dictionary, observers, generator):
        super().__init__(record_to_dictionary, observers)
        self.generator = generator

        self.name = "GeneratorDataSource: {}".format(self.generator.name)


    def listen_for_events(self):
        event = self.generator.next_sample()
        while event is not None:
            self.on_new_event(event)
            event = self.generator.next_sample()