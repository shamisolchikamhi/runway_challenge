import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder


class DataPrep:
    def __init__(self, dataset):
        self.dataset = dataset

    def formart_date(self):
        self.dataset['event_start_date'] = pd.to_datetime(self.dataset['event_start_date'], format="%Y/%m/%d")
        self.dataset['event_end_date'] = pd.to_datetime(self.dataset['event_end_date'], format="%Y/%m/%d")

        # return self.dataset

    def nan_values(self):
        self.dataset = self.dataset.fillna(0)

    def new_features(self):
        self.dataset['event_duration'] = (self.dataset['event_end_date'] - self.dataset[
            'event_start_date']) / np.timedelta64(1, 'D')
        self.dataset['year'] = self.dataset['event_start_date'].dt.year
        self.dataset['month'] = self.dataset['event_start_date'].dt.month

        self.dataset['rrsp'] = np.log(self.dataset['rrsp'])
        self.dataset['selling_price'] = np.log(self.dataset['selling_price'])
        self.dataset['qty_available'] = np.log(self.dataset['qty_available'])


    @staticmethod
    def categorical_encoding(dataset, columns = ['product_gender', 'product_type']):
        le = LabelEncoder()
        for i in columns:
            dataset[i] = le.fit_transform(dataset[i])


    def get_prediction_and_training_data(self, start='2019-04-01', end='2022-06-30'):
        prediction_rows = (self.dataset['event_start_date'] > start) & (self.dataset['event_start_date'] <= end)
        prediction_data = self.dataset.loc[prediction_rows]
        ml_rows = ~prediction_rows
        ml_data = self.dataset.loc[ml_rows]

        return [ml_data, prediction_data]

class DataExploration:
    def __init__(self, data):
        self.data = data

    def correlation(self, columns = None):
        corr_data = self.data.corr()

        if columns is None:
            corr_data = corr_data
        else:
            corr_data = corr_data[columns]

        return corr_data

    def histograms(self, column):
        plt.bar(column, "qty_invoiced", data=self.data, color="blue")
        plt.xlabel(column)
        plt.ylabel("quantity_invoiced")
        plt.show()










