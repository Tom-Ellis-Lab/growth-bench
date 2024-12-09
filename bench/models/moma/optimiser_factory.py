import abc
import dataclasses
import enum

import keras


class OptimiserType(enum.Enum):
    """Types of optimisers for the model."""

    ADAM = "Adam"
    SGD = "SGD"
    ADAGRAD = "Adagrad"
    # Add more optimiser types as needed


@dataclasses.dataclass
class OptimiserParams:
    """Configuration for the optimiser.

    Attributes
    ----------
    type : OptimiserType
        The type of the optimiser.
    params : dict
        The parameters for the optimiser.
    """

    type: OptimiserType
    params: dict


class OptimiserFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_optimiser(
        self, optimiser_params: "OptimiserParams"
    ) -> "keras.optimizers.Optimizer":
        """
        Create an optimiser based on the optimiser parameters.

        Parameters
        ----------
        optimiser_params : OptimiserParams
            The parameters for the optimiser.

        Returns
        -------
        keras.optimizers.Optimizer
            The optimiser.
        """
        pass


class OptimiserFactory(OptimiserFactoryInterface):
    def create_optimiser(
        self, optimiser_params: "OptimiserParams"
    ) -> "keras.optimizers.Optimizer":
        """
        Create an optimiser based on the optimiser parameters.

        Parameters
        ----------
        optimiser_params : OptimiserParams
            The parameters for the optimiser.

        Returns
        -------
        keras.optimizers.Optimizer
            The optimiser.
        """
        if optimiser_params.type == OptimiserType.ADAM:
            return keras.optimizers.Adam(**optimiser_params.params)
        elif optimiser_params.type == OptimiserType.SGD:
            return keras.optimizers.SGD(**optimiser_params.params)
        elif optimiser_params.type == OptimiserType.ADAGRAD:
            return keras.optimizers.Adagrad(**optimiser_params.params)
        else:
            raise ValueError(f"Unknown optimiser type: {optimiser_params.type}")
