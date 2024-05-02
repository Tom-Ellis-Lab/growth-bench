import dataclasses
from typing import Optional, Union

import numpy as np
import pandas as pd

from bench.models import strategy


@dataclasses.dataclass
class LassoParams(strategy.SetupParams):
    """
    Dataclass for LASSO parameterss
    """

    beta_gene_expression: np.ndarray
    beta_fluxes: np.ndarray
    intercept: float
    target_data: Optional[pd.Series] = None


class LassoStrategy(strategy.MultiOmicsStrategy):
    """
    Strategy for running LASSO on gene expression and flux data
    """

    def predict_task2(self, data: Union[np.ndarray, list[np.ndarray]]) -> np.ndarray:

        gene_expression_data = data[0]
        flux_data = data[1]
        data = np.concatenate((gene_expression_data, flux_data), axis=1)
        # Combine gene expression and flux coefficients
        beta = np.hstack([self._beta_gene_expression, self._beta_fluxes])

        # Compute predictions: X_new.dot(beta) + intercept
        result = data.dot(beta) + self.intercept
        return result

    def setup_strategy(self, params: LassoParams) -> None:
        self.beta_gene_expression = params.beta_gene_expression
        self.beta_fluxes = params.beta_fluxes
        self.intercept = params.intercept

    @property
    def beta_gene_expression(self) -> np.ndarray:
        return self._beta_gene_expression

    @beta_gene_expression.setter
    def beta_gene_expression(self, value: np.ndarray) -> None:
        self._beta_gene_expression = value

    @property
    def beta_fluxes(self) -> np.ndarray:
        return self._beta_fluxes

    @beta_fluxes.setter
    def beta_fluxes(self, value: np.ndarray) -> None:
        self._beta_fluxes = value

    @property
    def intercept(self) -> float:
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        self._intercept = value
