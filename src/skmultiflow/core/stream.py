from streamz import Stream

@Stream.register_api()
class predict(Stream):
    def __init__(self, upstream, model, **kwargs):
        self.model = model
        super().__init__(upstream, **kwargs)

    def update(self, X, who=None):
        return self._emit(self.model.predict(X))

@Stream.register_api()
class partial_fit(Stream):
    def __init__(self, upstream, model, **kwargs):
        self.model = model
        super().__init__(upstream, **kwargs)

    def update(self, x, who=None):
        X, y = X
        self.model.partial_fit(X, y)
        return self._emit(self.model)

@Stream.register_api()
class from_sklearn_pipeline(Stream):
    """
    take a pipeline and build a stream
    """
    pass  # TODO
