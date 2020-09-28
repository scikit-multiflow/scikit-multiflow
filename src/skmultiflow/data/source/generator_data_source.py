from skmultiflow.data.source.data_source import DataSource


class GeneratorDataSource(DataSource):
    """ GeneratorDataSource class.
    Provides a DataSource implementation, pulling data from a generator.

    Examples
    --------
    >>> import time
    >>> from skmultiflow.data.generator.sea_generator import SEAGenerator
    >>> from skmultiflow.data.source.generator_data_source import GeneratorDataSource
    >>> from skmultiflow.data.observer.buffer_data_event_observer import BufferDataEventObserver
    >>>
    >>>
    >>> def record_to_dictionary(record):
    >>>     if record is None:
    >>>         return None
    >>>     return record
    >>>
    >>>
    >>> # Setup an event observer and a data source
    >>> sea_generator = SEAGenerator(classification_function=2, random_state=112, balance_classes=False, noise_percentage=0.28)
    >>> buffer_data_event_observer = BufferDataEventObserver()
    >>> data_source = GeneratorDataSource(record_to_dictionary, [buffer_data_event_observer], sea_generator)
    >>> data_source.listen_for_events()
    >>>
    >>> # Wait until first event is received
    >>> while(len(buffer_data_event_observer.get_buffer())==0):
    >>>     time.sleep(0.100)  # 100ms
    >>>
    >>> first_event = buffer_data_event_observer.get_buffer()[0]
    >>> print('First event: X: {}, y: {}'.format(first_event['X'], first_event['y']))

    """
    def __init__(self, record_to_dictionary, observers, generator):
        super().__init__(record_to_dictionary, observers)
        self.generator = generator

        self.name = "GeneratorDataSource: {}".format(self.generator.name)

    def _prepare_for_use(self):
        pass

    def listen_for_events(self):
        #TODO: solve problem with infinite while in some better way
        X, y = self.generator.next_sample()
        event = {'X': X[0], 'y': y}
        count = 0
        while event is not None and count < 10:
            self.on_new_event(event)
            X, y = self.generator.next_sample()
            event = {'X': X[0], 'y': y}
            count += 1
