from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def create_ds(self):
        pass
