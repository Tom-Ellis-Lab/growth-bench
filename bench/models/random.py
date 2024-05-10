"""
This model is randomly guessing the correct answer by sampling a number from a normal distribution with mean 0 and standard deviation 1.
"""

from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np

from bench.models.strategy import Strategy

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class RandomNormal(Strategy):

    def predict_task1(self, data: pd.DataFrame) -> pd.DataFrame:
        data["prediction"] = np.random.normal(0, 0.01, data.shape[0])
        return data

    def predict_task2(self, data: List):
        raise NotImplementedError("This model does not support Task 2")
    
    def predict_task3(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("This model does not support Task 3")
    
    def predict_task4(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("This model does not support Task 4")
