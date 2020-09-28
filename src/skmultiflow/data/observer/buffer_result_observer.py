class BufferResultObserver():
    """ BufferResultObserver class.
    """

    def __init__(self, metrics, reporter, preserve_data=False, buffer_size=20, ):
        """ BufferResultObserver class constructor."""
        self.y_pred_buffer = []
        self.y_true_buffer = []

    def report(self, y_pred, y_true):
        self.y_pred_buffer = y_pred
        self.y_true_buffer = y_true
        
