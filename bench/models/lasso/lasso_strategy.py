import dataclasses
from typing import Optional, Union

import numpy as np

from bench.models import strategy


@dataclasses.dataclass
class LassoParams(strategy.SetupParams):
    """
    Dataclass for LASSO parameterss
    """

    beta_gene_expression: np.ndarray
    beta_fluxes: np.ndarray
    intercept: float
    target_data: Optional[np.ndarray] = None


class LassoStrategy(strategy.MultiOmicsStrategy):
    """
    Strategy for running LASSO on gene expression and flux data
    """

    def predict_task2(self, data: Union[np.ndarray, list[np.ndarray]]) -> np.ndarray:
        """
        Predict the target data for Task2

        Parameters
        ----------
        data : Union[np.ndarray, list[np.ndarray]]
            A list of two numpy arrays: flux data and gene expression data

        Returns
        -------
        np.ndarray
            The predicted target data
        """

        # The order is important: flux data first, then gene expression data
        flux_data = data[0]
        gene_expression_data = data[1]
        data = np.concatenate((gene_expression_data, flux_data), axis=1)

        result = (
            data @ np.hstack([self._beta_gene_expression, self._beta_fluxes])
            + self.intercept
        )
        return result

    def setup_strategy(self, params: LassoParams) -> None:
        """
        Setup the LASSO strategy

        Parameters
        ----------
        params : LassoParams
            A dataclass containing the LASSO parameters

        Returns
        -------
        None
        """
        self.beta_gene_expression = params.beta_gene_expression
        self.beta_fluxes = params.beta_fluxes
        self.intercept = params.intercept

    @property
    def beta_gene_expression(self) -> np.ndarray:
        """Get the beta coefficients for gene expression

        Returns
        -------
        np.ndarray
            The beta coefficients for gene expression
        """
        return self._beta_gene_expression

    @beta_gene_expression.setter
    def beta_gene_expression(self, value: np.ndarray) -> None:
        """Set the beta coefficients for gene expression

        Parameters
        ----------
        value : np.ndarray
            The beta coefficients for gene expression

        Returns
        -------
        None
        """
        self._beta_gene_expression = value

    @property
    def beta_fluxes(self) -> np.ndarray:
        """Get the beta coefficients for fluxes

        Returns
        -------
        np.ndarray
            The beta coefficients for fluxes
        """

        return self._beta_fluxes

    @beta_fluxes.setter
    def beta_fluxes(self, value: np.ndarray) -> None:
        """Set the beta coefficients for fluxes

        Parameters
        ----------
        value : np.ndarray
            The beta coefficients for fluxes

        Returns
        -------
        None
        """
        self._beta_fluxes = value

    @property
    def intercept(self) -> float:
        """Get the intercept

        Returns
        -------
        float
            The intercept
        """
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        """Set the intercept

        Parameters
        ----------
        value : float
            The intercept

        Returns
        -------
        None
        """
        self._intercept = value
