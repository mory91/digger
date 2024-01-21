import os
import numpy as np
from sklearn.model_selection import train_test_split
from model.abstract import Predictor
from constants import (
    CLASSIFICATION,
    REGRESSION,
    REGCLASS
)
import model.xgb.train as xgboost_learn


class Flux(Predictor):
    def __init__(self, flows_df, target_column='size'):
        self.flows_df = flows_df
        self.target_column = target_column
        tmp_path = 'tmp-data'
        self.target_file_name = "flows.csv"
        self.tmp_train_path = f"{tmp_path}/train"
        self.tmp_test_path = f"{tmp_path}/test"
        os.makedirs(self.tmp_train_path, exist_ok=True)
        os.makedirs(self.tmp_test_path, exist_ok=True)
        super().__init__()


class FluxRegression(Flux):
    def preprocess_data(self):
        return

    def train(self, split=0.5):
        flows_train, flows_test = train_test_split(
            self.flows_df, shuffle=False, test_size=split
        )

        flows_train.to_csv(
            f"{self.tmp_train_path}/{self.target_file_name}",
            index=False
        )
        flows_test.to_csv(
            f"{self.tmp_test_path}/{self.target_file_name}",
            index=False
        )

        r, m = xgboost_learn.catboost_learn(
            f"{self.tmp_train_path}/", f"{self.tmp_test_path}/",
            problem_type=REGRESSION
        )
        return r


class FluxClassifier(Flux):
    def __init__(self, flows_df, target_column='size', emthresh=3500):
        self.emthresh = emthresh
        super().__init__(flows_df, target_column)

    def preprocess_data(self):
        o = self.flows_df[self.target_column].values
        o = np.array(o > self.emthresh) * 1
        self.flows_df[self.target_column] = o

    def train(self, split=0.5):
        flows_train, flows_test = train_test_split(
            self.flows_df, shuffle=False, test_size=split
        )

        flows_train.to_csv(
            f"{self.tmp_train_path}/{self.target_file_name}",
            index=False
        )
        flows_test.to_csv(
            f"{self.tmp_test_path}/{self.target_file_name}",
            index=False
        )

        r, m = xgboost_learn.catboost_learn(
            f"{self.tmp_train_path}/", f"{self.tmp_test_path}/",
            problem_type=CLASSIFICATION, emthresh=self.emthresh
        )
        return r


class FluxRegClass(Flux):
    def __init__(self, flows_df, target_column='size', emthresh=3500):
        self.emthresh = emthresh
        super().__init__(flows_df, target_column)

    def preprocess_data(self):
        return

    def train(self, split=0.5):
        flows_train, flows_test = train_test_split(
            self.flows_df, shuffle=False, test_size=split
        )

        flows_train.to_csv(
            f"{self.tmp_train_path}/{self.target_file_name}",
            index=False
        )
        flows_test.to_csv(
            f"{self.tmp_test_path}/{self.target_file_name}",
            index=False
        )

        r, m = xgboost_learn.catboost_learn(
            f"{self.tmp_train_path}/", f"{self.tmp_test_path}/",
            problem_type=REGCLASS, emthresh=self.emthresh
        )
        return r
