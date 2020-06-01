import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessorMulti:
    def convert(self, csv_file_path, test_size):
        """

        :param csv_file_path: path of the csv file
        :param test_size: proportion of the dataset to include in the test split
        :return:
            X_train : {array-like, sparse matrix}, shape (n_samples, n_features) for train
            X_test : {array-like, sparse matrix}, shape (n_samples, n_features) for test
            y_train : {array-like, sparse matrix}, shape = (n_samples, y_targets) target values for train
            y_test : {array-like, sparse matrix}, shape = (n_samples, y_targets) target values for test
        """
        df = pd.read_csv(csv_file_path)
        df = self._data_processing(df)
        y = df[list(df.filter(regex='Class'))]
        X = df[df.columns.drop(list(df.filter(regex='Class')))]
        # X = df.drop(['Class'], axis=1)
        # multi manual
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def _data_processing(self, df):
        """

        :param df: {array-like, sparse matrix}, shape (n_samples, n_features)
        :return:
            df {array-like, sparse matrix}, shape (n_samples, n_features) after processing
        """
        for column in df.columns:
            if (df[column].dtype != np.int64 and df[column].dtype != np.float64):
                unique = np.unique(np.array(df[column]), return_counts=False)
                df[column] = df[column].replace(unique, list(range(0, len(unique))))
                df[column] = df[column].astype('category')
        return df