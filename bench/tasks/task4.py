import pandas as pd

from bench.tasks import task1


class Task4(task1.Task1):
    """
    Task 4 - comparing predictions against experimental data (e.g. growth rates)

    experimental data source: DUBHIR et al. https://www.embopress.org/doi/full/10.15252/msb.20145172
    """

    _data = pd.read_csv("data/tasks/task4/growth_rates_duibhir.csv")

    def predict(self) -> pd.DataFrame:
        """
        Predict the growth rate using task4

        Returns
        -------
        pd.DataFrame
            the predicted growth rate data
        """
        return self._strategy.predict_task4(self._data)
