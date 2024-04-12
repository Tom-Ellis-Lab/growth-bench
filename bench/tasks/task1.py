from bench.models.strategy import Strategy


class Task1:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

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

    def benchmark(self) -> dict:
        import pandas as pd
        import os

        # Load and process the data
        data = pd.read_csv("data/tasks/task1/growth_rate.csv")

        # rename column to true
        data.rename(
            columns={
                "hap a | growth (exponential growth rate) | standard | minimal complete | Warringer J~Blomberg A, 2003": "true",
                "orf": "knockout_gene_id",
            },
            inplace=True,
        )

        # Predict on dataset
        result = self._strategy.predict_task1(data)

        # Calculate performance metric (here, MSE and pearson correlation)
        from sklearn.metrics import mean_squared_error
        from scipy.stats import pearsonr

        results_notna = result.dropna()

        mse = mean_squared_error(results_notna["true"], results_notna["prediction"])
        pearson = pearsonr(results_notna["true"], results_notna["prediction"])[0]
        spearman = results_notna["true"].corr(
            results_notna["prediction"], method="spearman"
        )

        return {
            "mse": mse,
            "pearson": pearson,
            "spearman": spearman,
            "coverage": 1 - result["prediction"].isna().sum() / len(result),
        }
