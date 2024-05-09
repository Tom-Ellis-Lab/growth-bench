import os
from typing import Callable, Union
import pandas as pd
from cobra.io import read_sbml_model

from bench.models import strategy


class SimpleFBA(strategy.ConstraintBasedStrategy):
    """Simple Baseline FBA model
    https://www.ebi.ac.uk/biomodels/BIOMD0000001063#Overview

    Parameters
    ----------
    None

    Attributes
    ----------
    model_path: str
        path to the simplefba model
    model: cobra.Model
        simplefba model
    """
    def __init__(
            self,
            model_name: str = "simplefba",
            model_path: str = "data/models/simplefba/yeast-GEM.xml",
            gateway: Callable = read_sbml_model
    ) -> None:
        super().__init__(
            model_name=model_name, 
            model_path=model_path, 
            gateway=gateway
        )

    def _get_model_from_path(
        self, model_path: Union[str, None], gateway: Union[Callable, None]
    ) -> None:
        """
        For SimpleFBA, the file is too large to be stored in the repository.
        However, if the predictions exist, they are loaded from the file and 
        results can be computed.
        Else, raise an error.
        """
        predictions_path = "data/predictions/simplefba/growth_rate_predictions.csv"
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(f"Predictions file does not exist for SimpleFBA strategy: {predictions_path}")
    
    def predict_task2(self, data: list):
        raise NotImplementedError("This model does not support Task 2")
    
    def predict_task3(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("This model does not support Task 3")
    
    def setup_strategy(self, param: strategy.SetupParams) -> None:
        """
        This method should not be implemented in the Yeast9 model
        """
        raise NotImplementedError("This model does not support setup_strategy")
