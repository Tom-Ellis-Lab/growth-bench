"""
Model: MI-POGUE

This is a slightly adapted implementation of MI-POGUE from Wytock et al. (2018)
"""

from __future__ import annotations
from typing import List

from bench.models.strategy import Strategy

"""
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""


class Woytock2018(Strategy):

    def predict_task1(self, data: List) -> List:
        raise NotImplementedError("This model does not support Task 1")

    def predict_task2(self, data: List):
        raise NotImplementedError("This model does not support Task 2")
