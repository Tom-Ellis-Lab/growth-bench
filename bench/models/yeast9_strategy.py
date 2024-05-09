
import numpy as np

from bench.models import strategy
from temp_tools import matlab_model_gateway
from typing import Callable


class Yeast9Strategy(strategy.ConstraintBasedStrategy):
    """GEM taken from: https://github.com/SysBioChalmers/yeast-GEM/tree/main/model

    Parameters
    ----------
    None

    Attributes
    ----------
    model_path: str
        path to the yeast9 model
    model: cobra.Model
        yeast9 model
    """
    def __init__(
            self, 
            model_name: str ="yeast9", 
            model_path: str ="data/models/yeast9/yeast-GEM.mat",
            gateway: Callable= matlab_model_gateway.load_model_from_mat
    ) -> None:
        super().__init__(model_name=model_name, model_path=model_path, gateway=gateway)

    def predict_task2(self, data: list[np.ndarray]) -> np.ndarray:
        """
        This method should not be implemented in the Yeast9 model
        """
        raise NotImplementedError("This model does not support Task 2")


    def setup_strategy(self, param: strategy.SetupParams) -> None:
        """
        This method should not be implemented in the Yeast9 model
        """
        raise NotImplementedError("This model does not support setup_strategy")