import numpy as np

from bench.models import strategy
from bench.models.decrem import decrem_gateway


class Decrem(strategy.ConstraintBasedStrategy):
    """GEM taken from: https://github.com/lgyzngc/Decrem-1.0/tree/master/three%20reconstructed%20models

    Parameters
    ----------
    None

    Attributes
    ----------
    model_path: str
        path to the yeast9 model
    model: cobra.Model
        decrem model
    """

    def __init__(
            self, 
            model_name: str ="decrem", 
            model_path: str ="data/models/decrem/Yeast_linear_model.mat",
            gateway: callable = decrem_gateway.load_matlab_model
    ) -> None:
        super().__init__(model_name=model_name, model_path=model_path, gateway=gateway)

    def predict_task2(self, data: list[np.ndarray]) -> np.ndarray:
        """
        This method should not be implemented in the Decrem model
        """
        raise NotImplementedError("This model does not support Task 2")


    def setup_strategy(self, param: strategy.SetupParams) -> None:
        """
        This method should not be implemented in the Decrem model
        """
        raise NotImplementedError("This model does not support setup_strategy")