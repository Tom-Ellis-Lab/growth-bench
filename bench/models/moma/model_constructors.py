import abc
import dataclasses
import enum

import keras


class LayerType(enum.Enum):
    """
    Types of layers in the model.

    Attributes
    ----------
    DENSE : str
        Dense layer.
    CONV2D : str
        Conv2D layer.
    FLATTEN : str
        Flatten layer.
    """

    DENSE = "Dense"
    DROPOUT = "Dropout"
    CONV2D = "Conv2D"
    FLATTEN = "Flatten"
    # Add more layer types as needed


@dataclasses.dataclass
class LayerConfig:
    """
    Configuration for a layer in the model.

    Attributes
    ----------
    type : LayerType
        The type of the layer.
    params : dict
        The parameters for the layer.
    """

    type: LayerType
    params: dict


@dataclasses.dataclass
class ModelParams:
    """
    Parameters for the model.

    Attributes
    ----------
    name : str
        The name of the model.
    architecture : list[LayerConfig]
        The architecture of the model.
    """

    name: str
    architecture: list[LayerConfig]
    input_size: int


@dataclasses.dataclass
class MultViewModelParams:
    """
    Parameters for the multi view model.

    Attributes
    ----------
    name : str
        The name of the model.
    architecture : list[LayerConfig]
        The architecture of the model.
    models : dict[int, keras.Model]
        The pairs of input size and model
    """

    name: str
    architecture: list[LayerConfig]
    models: dict[int, keras.Model]

    def __post_init__(self):
        self._verify_input_sizes()

    def _verify_input_sizes(self):
        """Verify that the input sizes of the models match the input sizes in the model parameters."""
        for input_size, model in self.models.items():
            if model.input_shape[1] != input_size:
                raise ValueError(
                    f"Input size of model '{model.name}' does not match the input size in the model parameters."
                )


class LayerFactory:

    @staticmethod
    def create_layer(layer_config: LayerConfig) -> keras.layers.Layer:
        """Create a layer based on the layer configuration.

        Parameters
        ----------
        layer_config : LayerConfig
            The layer configuration.

        Returns
        -------
        keras.layers.Layer
            The layer.
        """
        layer_type = layer_config.type
        layer_params = layer_config.params

        if layer_type == LayerType.DENSE:
            return keras.layers.Dense(**layer_params)
        elif layer_type == LayerType.DROPOUT:
            return keras.layers.Dropout(**layer_params)
        elif layer_type == LayerType.CONV2D:
            return keras.layers.Conv2D(**layer_params)
        elif layer_type == LayerType.FLATTEN:
            return keras.layers.Flatten(**layer_params)
        # Add more layer types as needed
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")


class ModelConstructor(abc.ABC):
    @abc.abstractmethod
    def construct(self, model_params: ModelParams) -> keras.Model:
        """
        Construct a model based on the model parameters.

        Parameters
        ----------
        model_params : ModelParams
            The parameters for the model.

        Returns
        -------
        Model
            The model.
        """
        pass


class SingleViewModelConstructor(ModelConstructor):

    def construct(self, model_params: ModelParams) -> keras.Model:
        """Construct a single view model based on the model parameters.

        Parameters
        ----------
        model_params : ModelParams
            The parameters for the model.

        Returns
        -------
        keras.Model
            The model.
        """
        input = keras.layers.Input(shape=(model_params.input_size,))
        current_layer = input
        for layer_config in model_params.architecture:
            layer = LayerFactory.create_layer(layer_config)
            current_layer = layer(current_layer)

        model = keras.Model(inputs=input, outputs=current_layer, name=model_params.name)
        return model


class MultiViewModelConstructor(ModelConstructor):

    def construct(self, model_params: MultViewModelParams) -> keras.Model:
        """Construct a multi view model based on the model parameters.

        Parameters
        ----------
        model_params : MultViewModelParams
            The parameters for the model.

        Returns
        -------
        keras.Model
            The model.
        """

        inputs = [
            keras.layers.Input(shape=(input_size,))
            for input_size in model_params.models.keys()
        ]
        model_outputs = [
            model(input_layer)
            for input_layer, model in zip(inputs, model_params.models.values())
        ]

        current_layer = keras.layers.Concatenate()(model_outputs)
        for layer_config in model_params.architecture:
            layer = LayerFactory.create_layer(layer_config)
            current_layer = layer(current_layer)

        model = keras.Model(
            inputs=inputs, outputs=current_layer, name=model_params.name
        )

        return model
