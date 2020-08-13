import math

import scipy.stats as stats

import numpy as np


class IfnMetaLearning:

    def __init__(self, alpha, number_of_classes):
        """
        Parameters
        ----------
        alpha : float
            Significance level
        number_of_classes : int
            The number of classes in the target.

        """
        if 0 <= alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError("Enter a valid alpha between 0 to 1")
        if 1 < number_of_classes:
            self.classes = number_of_classes
        else:
            raise ValueError("Enter number of classes bigger than 1")
        self.window = 0

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        self._window = value

    def calculate_Wint(self, Pe):
        """ This function calculate the size of the initial window based on formula 7 in
            "https://content.iospress.com/articles/intelligent-data-analysis/ida00083".

        Parameters
        ----------
        Pe : float
            Maximum allowable prediction error of the model.


        Returns
        -------
            The size of the initial window.
        """

        if Pe < 0:
            raise ValueError("Pe shouldn't be negative")

        chi2_alpha = stats.chi2.ppf(self.alpha, self.classes - 1)
        entropy_Pe = stats.entropy([Pe, 1 - Pe], base=2)

        denominator = 2 * np.log(2) * \
                      (math.log(self.classes, 2) - entropy_Pe - Pe * math.log(self.classes - 1, 2))

        if denominator == 0:
            self.window = 0
        else:
            self.window = int(chi2_alpha / denominator)

        return self.window

    def calculate_new_window(self, NI, T, Etr):
        """ This function calculate the size of the new window based on formula 8 in
            "https://content.iospress.com/articles/intelligent-data-analysis/ida00083".

        Parameters
        ----------
        NI : int
           The number of values (or discretized intervals) for the first attribute in the info-fuzzy network.
           This number should be bigger than one.
        T : float
            The distribution of the target class in the info-fuzzy network.
            a number between 0 and 1.
        Etr : float
            The error rate of the training set.
            a number between 0 and 1.


        Returns
        -------
            The size of the initial window.
        """
        if NI < 2:
            raise ValueError("Number of classes should be a least 2")
        if T < 0 or T > 1:
            raise ValueError("Target distribution should be between 0 and 1")
        if Etr < 0 or Etr > 1:
            raise ValueError("Error rate of the training set should be between 0 and 1")

        chi2_alpha = stats.chi2.ppf(self.alpha, (NI - 1) * (self.classes - 1))
        entropy_T = stats.entropy([T, 1 - T], base=2)
        entropy_Etr = stats.entropy([Etr, 1 - Etr], base=2)

        denominator = 2 * np.log(2) * \
                      (entropy_T - entropy_Etr - Etr * math.log(self.classes - 1, 2))

        if denominator == 0:
            self.window = 0
        else:
            self.window = int(chi2_alpha / denominator)
            if self.window <= 0:
                self.window = 0

        return self.window

    def _calculate_var_diff(self, Etr, Eval, add_count):
        """ This function calculate the variance of the difference between the two error rates (Etr and Eval)
            based on formula 9 in "https://content.iospress.com/articles/intelligent-data-analysis/ida00083".

        Parameters
        ----------
        Etr : float
            The error rate of the training set.
        Eval : float
            The error rate of the validation set.
        add_count  : int/float
            The number of validation examples.

        Returns
        -------
            The variance between the Etr and Eval.
        """

        return (Etr * (1 - Etr) / self.window) + (Eval * (1 - Eval) / add_count)

    def get_max_diff(self, Etr, Eval, add_count):
        """ This function calculate the maximum difference between the error rates, at the
            99% confidence level based on formula 10 in
            "https://content.iospress.com/articles/intelligent-data-analysis/ida00083".

        Parameters
        ----------
        Etr : float
            The error rate of the training set.
        Eval : float
            The error rate of the validation set.
        add_count  : int/float
            The number of validation examples.

        Returns
        -------
            The maximum difference between the Etr and Eval.
        """
        if Etr < 0 or Etr > 1:
            raise ValueError("Error rate of the training set should be between 0 and 1")
        if Eval < 0 or Eval > 1:
            raise ValueError("Error rate of the validation set should be between 0 and 1")
        if add_count < 1:
            raise ValueError("The number of validation examples shouldn't by negative")

        return stats.norm.ppf(0.99) * self._calculate_var_diff(Etr, Eval, add_count)
