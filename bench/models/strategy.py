import abc
import dataclasses

import numpy as np
import pandas as pd


class Strategy(abc.ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abc.abstractmethod
    def predict_task1(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def predict_task2(self, data: list):
        pass


@dataclasses.dataclass
class SetupParams:
    """Dataclass for setting up the strategy.

    Parameters
    ----------
    gene_expression_data : np.ndarray
        The gene expression data.
    flux_data : np.ndarray
        The flux data.
    target_data : Optional[np.ndarray]
        The target data. Default is None.
    """

    gene_expression_data: np.ndarray
    flux_data: np.ndarray


class MultiOmicsStrategy(Strategy):

    def predict_task1(self, data: pd.DataFrame) -> None:
        """
        This method should not be implemented in the MultiOmicsStrategy class
        """
        raise NotImplementedError("This model does not support Task 1")

    @abc.abstractmethod
    def predict_task2(self, data: list[np.ndarray]) -> np.ndarray:
        pass

    @abc.abstractmethod
    def setup_strategy(self, param: SetupParams) -> None:
        pass
