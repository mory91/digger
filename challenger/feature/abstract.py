from abc import ABC, abstractmethod


class Feature(ABC):
    @abstractmethod
    def create_feature_df(self):
        pass
