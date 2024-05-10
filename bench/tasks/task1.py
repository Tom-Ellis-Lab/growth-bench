import abc

import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from bench.models.strategy import Strategy


class Task(abc.ABC):
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy
        self._predict = None

    @property
    def strategy(self) -> Strategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    @abc.abstractmethod
    def benchmark(self) -> dict:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        pass

    @abc.abstractmethod
    def predict(self) -> pd.DataFrame:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        pass


class Task1(Task):
    """
    Task 1 - comparing predictions against experimental data (e.g. growth rates)

    experimental data source: WARRINGER & BLOMBERG https://www.yeastgenome.org/reference/S000140659
    """

    _data = pd.read_csv("data/tasks/task1/growth_rates_warringer.csv")

    def __init__(self, strategy: Strategy) -> None:
        """
        Parameters
        ----------
        strategy: Strategy
            the strategy to use for the task
        """
        self._strategy = strategy

    def benchmark(self) -> dict[str, float]:
        """
        Calculate the performance metrics for the task

        Returns
        -------
        dict
            the metrics for the task (MSE, pearson, spearman, coverage)
        """
        result = self.predict()

        results_notna = result.dropna()

        mse = mean_squared_error(results_notna["true"], results_notna["prediction"])
        pearson = pearsonr(results_notna["true"], results_notna["prediction"])[0]
        spearman = results_notna["true"].corr(
            results_notna["prediction"], method="spearman"
        )

        return {
            "mse": float(mse),
            "pearson": pearson,
            "spearman": spearman,
            "coverage": 1 - result["prediction"].isna().sum() / len(result),
        }

    def predict(self) -> pd.DataFrame:
        """
        Predict the growth rate using task1

        Returns
        -------
        pd.DataFrame
            the predicted growth rate data
        """
        return self._strategy.predict_task1(self._data)
