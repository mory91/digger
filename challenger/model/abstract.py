from abc import ABC, abstractmethod


class Predictor(ABC):
    def __init__(self):
        self.preprocess_data()

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def train(self):
        pass
