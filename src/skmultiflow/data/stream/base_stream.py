from abc import ABCMeta
from skmultiflow.data.observer import EventObserver
import numpy as np
import warnings

class Stream(EventObserver, metaclass=ABCMeta):
    """ Base Stream class.

    This abstract class defines the minimum requirements of a stream,
    so that it can work along other modules in scikit-multiflow.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """
    def __init__(self):
        self.source = None
        self.n_targets = 0
        self.n_features = 0
        self.n_num_features = 0
        self.n_cat_features = 0
        self.n_classes = 0
        self.cat_features_idx = []
        self.feature_names = None
        self.target_names = None
        self.target_values = None
        self.name = None

    @property
    def n_features(self):
        """ Retrieve the number of features.

        Returns
        -------
        int
            The total number of features.

        """
        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        """ Set the number of features

        """
        self._n_features = n_features

    @property
    def n_cat_features(self):
        """ Retrieve the number of integer features.

        Returns
        -------
        int
            The number of integer features in the stream.

        """
        return self._n_cat_features

    @n_cat_features.setter
    def n_cat_features(self, n_cat_features):
        """ Set the number of integer features

        Parameters
        ----------
        n_cat_features: int
        """
        self._n_cat_features = n_cat_features

    @property
    def n_num_features(self):
        """ Retrieve the number of numerical features.

        Returns
        -------
        int
            The number of numerical features in the stream.

        """
        return self._n_num_features

    @n_num_features.setter
    def n_num_features(self, n_num_features):
        """ Set the number of numerical features

        Parameters
        ----------
        n_num_features: int

        """
        self._n_num_features = n_num_features

    @property
    def n_targets(self):
        """ Retrieve the number of targets

        Returns
        -------
        int
            the number of targets in the stream.
        """
        return self._target_idx

    @n_targets.setter
    def n_targets(self, n_targets):
        """ Set the number of targets

        Parameters
        ----------
        n_targets: int
        """
        self._target_idx = n_targets

    @property
    def target_values(self):
        """ Retrieve all target_values in the stream for each target.

        Returns
        -------
        list
            list of lists of all target_values for each target
        """
        return self._target_values

    @target_values.setter
    def target_values(self, target_values):
        """ Set the list for all target_values in the stream.

        Parameters
        ----------
        target_values
        """
        self._target_values = target_values

    @property
    def feature_names(self):
        """ Retrieve the names of the features.

        Returns
        -------
        list
            names of the features
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        """ Set the name of the features in the stream.

        Parameters
        ----------
        feature_names: list
        """
        self._feature_names = feature_names

    @property
    def target_names(self):
        """ Retrieve the names of the targets

        Returns
        -------
        list
            the names of the targets in the stream.
        """
        return self._target_names

    @target_names.setter
    def target_names(self, target_names):
        """ Set the names of the targets in the stream.

        Parameters
        ----------
        target_names: list

        """
        self._target_names = target_names

    @property
    def target_idx(self):
        """
        Get the number of the column where Y begins.

        Returns
        -------
        int:
            The number of the column where Y begins.
        """
        return self._target_idx

    @target_idx.setter
    def target_idx(self, target_idx):
        """
        Sets the number of the column where Y begins.

        Parameters
        ----------
        target_idx: int
        """

        self._target_idx = target_idx

    @property
    def cat_features_idx(self):
        """
        Get the list of the categorical features index.

        Returns
        -------
        list:
            List of categorical features index.

        """
        return self._cat_features_idx

    @cat_features_idx.setter
    def cat_features_idx(self, cat_features_idx):
        """
        Sets the list of the categorical features index.

        Parameters
        ----------
        cat_features_idx:
            List of categorical features index.
        """

        self._cat_features_idx = cat_features_idx

    def update(self, event):
        """ Process next event in stream

        Parameters
        ----------
        event:
            dictionary with event data, to be processed
        """
        return self.process_entry(event)

    def process_entry(self, entry):
        """ Reads the data provided by the user and separates the features and targets.
            Performs basic checks for data consistency
        """
        X = entry['X']
        y = entry['y']
        self.check_data_consistency(X, self.allow_nan)
        self.check_data_consistency(y, self.allow_nan)
        _, self.n_features = X.shape
        self.n_num_features = self.n_features - self.n_cat_features
        return self.process_additional_entry_data(entry)

    def process_additional_entry_data(self, entry):
        """ Should override, if additional fields are considered for
        the entry or same fields require more processing.
        """
        return entry['X'], entry['y']

    def is_restartable(self):
        """
        Determine if the stream is restartable.

        Returns
        -------
        Bool
            True if stream is restartable.

        """
        return self.source.is_restartable()

    def restart(self):
        """  Restart the stream. """
        self.source.restart()

    def is_numeric_array(self, array):
        """Checks if the dtype of the array is numeric.

        Booleans, unsigned integer, signed integer, floats and complex are
        considered numeric.

        Parameters
        ----------
        array : `numpy.ndarray`-like
            The array to check.

        Returns
        -------
        is_numeric : `bool`
            True if it is a recognized numerical and False if object or
            string.
        """
        numerical_dtype_kinds = {'b',  # boolean
                                 'u',  # unsigned integer
                                 'i',  # signed integer
                                 'f',  # floats
                                 'c'}  # complex
        try:
            return array.dtype.kind in numerical_dtype_kinds
        except AttributeError:
            # in case it's not a numpy array it will probably have no dtype.
            return np.asarray(array).dtype.kind in numerical_dtype_kinds

    def check_data_consistency(self, row, allow_nan=False):
        """
        Check data consistency with respect to scikit-multiflow assumptions:
        * Only numeric data types are used.
        * Missing values are, in general, not supported.
        Parameters
        ----------
        raw_data_frame: pandas.DataFrame
            The data frame containing the data to check.
        allow_nan: bool, optional (default=False)
            If True, allows NaN values in the data. Otherwise, an error is raised.
        """

        if (not self.is_numeric_array(row)):
            # scikit-multiflow assumes that data is numeric
            raise ValueError('Non-numeric data found:\n {}'
                             'scikit-multiflow only supports numeric data.'
                             .format(row.dtypes))

        if np.isnan(row).any():
            if not allow_nan:
                raise ValueError("NaN values found. Missing values are not fully supported.\n"
                                 "You can deactivate this error via the 'allow_nan' option.")
            else:
                warnings.warn("NaN values found.", UserWarning)


    def get_data_info(self):
        """ Retrieves minimum information from the stream

        Used by evaluator methods to id the stream.

        The default format is: 'Stream name - n_targets, n_classes, n_features'.

        Returns
        -------
        string
            Stream data information

        """
        return self.name + " - {} target(s), {} classes, {} features".format(self.n_targets,
                                                                             self.n_classes,
                                                                             self.n_features)
