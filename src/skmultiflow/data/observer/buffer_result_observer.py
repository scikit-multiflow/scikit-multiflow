class BufferResultObserver():
    """ BufferResultObserver class.
    """

    def __init__(self):
        """ BufferResultObserver class constructor."""
        self.y_pred_buffer = []
        self.y_true_buffer = []

    # TODO: add policies, regarding we should preserve or discard data
    # TODO: compute metrics and report them somewhere: write to file, print to println, use them to display some graph, etc
    def report(self, y_pred, y_true):
        self.y_pred_buffer = y_pred
        self.y_true_buffer = y_true
        print("We just reported some results :)")
