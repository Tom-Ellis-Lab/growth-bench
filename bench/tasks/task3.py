import pandas as pd
from scipy.stats import pearsonr
from bench.tasks import task1


class Task3(task1.Task):
    """Task 3 - comparing predictions against experimental data (e.g. growth rates)

    experimental data source: RALSER https://www.sciencedirect.com/science/article/pii/S0092867423003008?via%3Dihub

    The growth rate was measured on 3 different media: SC, SM, YPD
    """

    _data = pd.read_csv("data/tasks/task3/yeast5k_growthrates_byORF.csv")
    _media = ["SC", "SM", "YPD"]

    def benchmark(self) -> dict:
        """Calculate the performance metrics for the task

        Returns
        -------
        dict
            the metrics for each medium (SC, SM, YPD)
        """

        # Predict on dataset
        result = self.predict()
        results_notna = result.dropna()
        result = {}
        for medium in self._media:
            metrics = self.get_metrics_for_medium(data=results_notna, medium_col=medium)
            result[medium] = metrics

        return result

    def predict(self) -> pd.DataFrame:
        """Predict the growth rate using task3

        Returns
        -------
        pd.DataFrame
            the predicted growth rate data
        """
        result = self._strategy.predict_task3(self._data)
        return result

    def get_metrics_for_medium(
        self, data: pd.DataFrame, medium_col: str
    ) -> dict[str, float]:
        """Get the metrics for the given medium (e.g. SC, SM, YPD)

        Parameters
        ----------
        data: pd.DataFrame
            the data to predict on

        medium_col: str
            the column name for the medium

        Returns
        -------
        dict[str, float]
            the metrics for the given medium (pearson, spearman, coverage)
        """
        pearson = pearsonr(data[medium_col], data["prediction"])[0]
        spearman = data[medium_col].corr(data["prediction"], method="spearman")

        return {
            "pearson": pearson,
            "spearman": spearman,
            "coverage": 1 - data["prediction"].isna().sum() / len(data),
        }
