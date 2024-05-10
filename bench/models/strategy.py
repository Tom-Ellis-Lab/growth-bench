import abc
import dataclasses
import multiprocessing
import os
import tqdm
from typing import Union, Callable, Optional

import cobra
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

    @abc.abstractmethod
    def predict_task3(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def predict_task4(self, data: pd.DataFrame) -> pd.DataFrame:
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


class ConstraintBasedStrategy(Strategy):
    """
    The ConstraintBasedStrategy abstract class defines the interface for the ConstraintBased
    models. This class should be inherited by all ConstraintBased models.

    ConstraintBased models are the Constraint-Based Models (e.g, SimpleFBA, Yeast9, Decrem).
    These models are benchmarked through gene essentiality analysis.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        gateway: Optional[Callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_name: str
            name of the model, must match with the model's name in the data/models folder
        model_path: str
            path to the model
        gateway: callable
            function to load the model, must match with the model's format
            (e.g. decrem_gateway.load_model_from_mat, matlab_model_gateway.load_model_from_mat)
        """
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.model = self._get_model_from_path(
            model_path=self.model_path, gateway=gateway
        )

    def _predict_single_growth_rate(
        self, index_row_tuple
    ) -> tuple[int, Union[float, None]]:
        """
        Parameters
        ----------
        index_row_tuple: tuple[int, pd.Series]
            index of the row and the row of the DataFrame

        Returns
        -------
        tuple[int, Union[float, None]]
            index of the row and the predicted growth rate
        """
        index, row = index_row_tuple
        try:
            gene_id = row["knockout_gene_id"]
            gene = self.model.genes.get_by_id(gene_id)
            gene.knock_out()
            solution = self.model.optimize()
            return index, solution.objective_value
        except:
            return index, None

    def predict_task1(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        data: pd.DataFrame
            gene knockout data

        Returns
        -------
        pd.DataFrame
            predicted growth rate data

        Predict the growth rate given the gene knockout data
        """
        results_path = "data/predictions/" + self.model_name + "/task1_results.csv"
        if os.path.exists(results_path):
            print("\tUsing cached results")
            data = pd.read_csv(results_path)
            return data
        data = self._predict_growth_rates(data=data, results_path=results_path)
        return data

    def _predict_growth_rates(self, data: pd.DataFrame, results_path) -> pd.DataFrame:
        """
        Parameters
        ----------
        data: pd.DataFrame
            gene knockout data

        results_path: str
            path to save the results

        Returns
        -------
        pd.DataFrame
            predicted growth rate data

        Predict the growth rate given the gene knockout data
        """
        predictions_path = (
            "data/predictions/" + self.model_name + "/growth_rate_predictions.csv"
        )

        if os.path.exists(predictions_path):
            print("\tUsing cached predictions")
            results = pd.read_csv(predictions_path)
            results = [(row.Index, row.prediction) for row in results.itertuples()]

        else:
            print(f"\tUsing {multiprocessing.cpu_count()} cores")
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = list(
                    tqdm.tqdm(
                        pool.imap(self._predict_single_growth_rate, data.iterrows()),
                        total=len(data),
                    )
                )

        for index, result in results:
            data.at[index, "prediction"] = result

        print(f"Could not build model for {data['prediction'].isna().sum()} genes")
        data.to_csv(results_path, index=False)
        return data

    def _get_model_from_path(
        self, model_path: Union[str, None], gateway: Union[Callable, None]
    ) -> cobra.Model:
        """
        This method should not be implemented in the ConstraintBasedStrategy class
        """
        if model_path is None:
            raise NotImplementedError("Model path is not defined")
        if gateway is not None:
            result = gateway(model_path)
        else:
            raise ValueError("Gateway function is not defined")
        return result

    @abc.abstractmethod
    def predict_task2(self, data: np.ndarray) -> np.ndarray:
        pass

    def predict_task3(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the growth rate given the gene knockout data

        Parameters
        ----------
        data: pd.DataFrame
            gene knockout data

        Returns
        -------
        pd.DataFrame
            predicted growth rate data
        """
        results_path = "data/predictions/" + self.model_name + "/task3_results.csv"
        if os.path.exists(results_path):
            print("\tUsing cached results")
            data = pd.read_csv(results_path)
            return data
        data = self._predict_growth_rates(data=data, results_path=results_path)
        return data

    def predict_task4(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict the growth rate given the gene knockout data
        Parameters
        ----------
        data: pd.DataFrame
            gene knockout data

        Returns
        -------
        pd.DataFrame
            predicted growth rate data
        """
        results_path = "data/predictions/" + self.model_name + "/task4_results.csv"
        if os.path.exists(results_path):
            print("\tUsing cached results")
            data = pd.read_csv(results_path)
            return data
        data = self._predict_growth_rates(data=data, results_path=results_path)
        return data

    @abc.abstractmethod
    def setup_strategy(self, param: SetupParams) -> None:
        pass


class MultiOmicsStrategy(Strategy):
    """The MultiOmicsStrategy abstract class defines the interface for the MultiOmics
    models. This class should be inherited by all MultiOmics models.

    MultiOmics models are models that use both gene expression and flux data to predict
    the growth rate of an organism (e.g MOMA, Lasso).
    """

    def predict_task1(self, data: pd.DataFrame) -> None:
        """
        This method should not be implemented in the MultiOmicsStrategy class
        """
        raise NotImplementedError("This model does not support Task 1")

    @abc.abstractmethod
    def predict_task2(self, data: list[np.ndarray]) -> np.ndarray:
        pass

    def predict_task3(self, data: pd.DataFrame) -> None:
        """
        This method should not be implemented in the MultiOmicsStrategy class
        """
        raise NotImplementedError("This model does not support Task 3")

    def predict_task4(self, data: pd.DataFrame) -> None:
        """
        This method should not be implemented in the MultiOmicsStrategy class
        """
        raise NotImplementedError("This model does not support Task 4")

    @abc.abstractmethod
    def setup_strategy(self, param: SetupParams) -> None:
        pass
