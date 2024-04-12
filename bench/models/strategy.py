from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class Strategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def predict_task1(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def predict_task2(self, data: List):
        pass
