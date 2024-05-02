import dataclasses
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Callable, Optional

from bench.models import strategy


@dataclasses.dataclass
class MomaParams(strategy.SetupParams):
    """Dataclass for MOMA parameters.

    Parameters
    ----------
    multi_model_weights_path : str
        The path to the multi-model weights.
    gene_expression_weights_path : str
        The path to the gene expression model weights.
    flux_weights_path : str
        The path to the flux model weights.
    learning_rate : float
        The learning rate.
    epochs : int
        The number of epochs.
    momentum : float
        The momentum.
    neurons_single_view : int
        The number of neurons for the single view models.
    neurons_multi_view : int
        The number of neurons for the multi-view model.

    """

    multi_model_weights_path: str
    gene_expression_weights_path: str
    flux_weights_path: str
    learning_rate: float
    epochs: int
    momentum: float
    neurons_single_view: int
    neurons_multi_view: int
    target_data: Optional[np.ndarray] = None


class MomaStrategy(strategy.MultiOmicsStrategy):

    def __init__(self) -> None:
        self._multi_model: tf.keras.Model = None
        self._gene_expression_model: tf.keras.Model = None
        self._flux_model: tf.keras.Model = None
        self._multi_model_weights_path: Optional[str] = None
        self._gene_expression_weights_path: Optional[str] = None
        self._flux_weights_path: Optional[str] = None
        self._gene_expression_data: Optional[np.ndarray] = None
        self._flux_data: Optional[np.ndarray] = None

    def predict_task2(self, data: list[np.ndarray]) -> np.ndarray:
        """Predict the target data using the multi-view model.

        Parameters
        ----------
        data : np.ndarray
            The data to predict.

        Returns
        -------
        np.ndarray
            The predicted target data.
        """
        if not self._multi_model_weights_path:
            raise ValueError("Multi model weights have not been set.")

        results = self._multi_model.predict(data)
        return results

    def setup_strategy(self, params: MomaParams) -> None:
        """Set up the strategy.

        Parameters
        ----------
        params : MomaParams
            The parameters to set up the strategy.

        Returns
        -------
        None
        """
        self.set_flux_data(data=params.flux_data)
        self.set_gene_expression_data(data=params.gene_expression_data)
        self.set_gene_expression_model(
            build_model=self.initialise_single_view_model,
            neurons=params.neurons_single_view,
        )
        self.set_flux_model(
            build_model=self.initialise_single_view_model,
            neurons=params.neurons_single_view,
        )
        self.load_gene_expression_weights_from_path(
            path=params.gene_expression_weights_path
        )
        self.load_flux_weights_from_path(path=params.flux_weights_path)

        self.set_multi_model(
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            momentum=params.momentum,
            neurons=params.neurons_multi_view,
            build_model=self.initialise_multi_view_model,
        )
        self.load_multi_view_weights_from_path(path=params.multi_model_weights_path)

    def load_multi_view_weights_from_path(self, path: str) -> None:
        """Load the multi-view model weights from a path.

        Parameters
        ----------
        path : str
            The path to the model weights.

        Returns
        -------
        None
        """
        if not self._multi_model:
            raise ValueError("Model has not been set.")

        if not os.path.exists(path):
            raise ValueError(
                "Model weights path does not exist. Check the path. if the weights are not saved, train the model first."
            )

        self._multi_model.load_weights(path)
        self._multi_model_weights_path = path

    def set_multi_model(
        self,
        learning_rate: float,
        epochs: int,
        momentum: float,
        neurons: int,
        build_model: Callable[..., object],
    ):
        """Set the multi-view model.

        Parameters
        ----------
        learning_rate : float
            The learning rate.
        epochs : int
            The number of epochs.
        momentum : float
            The momentum.
        neurons : int
            The number of neurons.

        Returns
        -------
        None
        """
        if not self._gene_expression_model or not self._flux_model:
            raise ValueError("Models have not been set.")

        if (
            self._gene_expression_data is None
            or self._gene_expression_data.shape[0] == 0
        ) or (self._flux_data is None or self._flux_data.shape[0] == 0):
            raise ValueError("Data has not been set.")

        self._multi_model = build_model(
            gene_expression_data_dim=self._gene_expression_data.shape[1],
            flux_data_dim=self._flux_data.shape[1],
            learning_rate=learning_rate,
            epochs=epochs,
            momentum=momentum,
            neurons=neurons,
            gene_expression_model=self._gene_expression_model,
            flux_model=self._flux_model,
        )

    @staticmethod
    def initialise_multi_view_model(
        flux_data_dim: int,
        gene_expression_data_dim: int,
        learning_rate: float,
        epochs: int,
        momentum: float,
        neurons: int,
        flux_model: tf.keras.Model,
        gene_expression_model: tf.keras.Model,
    ) -> tf.keras.Model:
        """Initialise the multi-view model.

        Parameters
        ----------
        gene_expression_data_dim : int
            The dimension of the gene expression data.
        flux_data_dim : int
            The dimension of the flux data.
        learning_rate : float
            The learning rate.
        epochs : int
            The number of epochs.
        momentum : float
            The momentum.
        neurons : int
            The number of neurons.
        gene_expression_model : tf.keras.Model
            The gene expression model.
        flux_model : tf.keras.Model
            The flux model.

        Returns
        -------
        tf.keras.Model
            The multi-view model.
        """

        input_1 = tf.keras.layers.Input(shape=(flux_data_dim,))
        input_2 = tf.keras.layers.Input(shape=(gene_expression_data_dim,))

        combined_layer = tf.keras.layers.Concatenate()(
            [flux_model(input_1), gene_expression_model(input_2)]
        )
        combined_layer = tf.keras.layers.Dense(
            neurons,
            activation="sigmoid",
            kernel_constraint=tf.keras.constraints.max_norm(3),
            name="last_hidden",
        )(combined_layer)

        predictions = tf.keras.layers.Dense(1, activation="linear")(combined_layer)
        result = tf.keras.Model(inputs=[input_1, input_2], outputs=predictions)

        sgd = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            weight_decay=learning_rate / epochs,
            momentum=momentum,
        )
        result.compile(
            loss="mean_squared_error", optimizer=sgd, metrics=["mean_absolute_error"]
        )
        return result

    def load_gene_expression_weights_from_path(self, path: str) -> None:
        """Load the gene expression model weights from a path.

        Parameters
        ----------
        path : str
            The path to the model weights.

        Returns
        -------
        None
        """
        if not self._gene_expression_model:
            raise ValueError("Model has not been set.")
        self._gene_expression_model.load_weights(path)

    def load_flux_weights_from_path(self, path: str) -> None:
        """Load the flux model weights from a path.

        Parameters
        ----------
        path : str
            The path to the model weights.

        Returns
        -------
        None
        """
        if not self._flux_model:
            raise ValueError("Model has not been set.")
        self._flux_model.load_weights(path)

    def set_gene_expression_model(
        self,
        build_model: Callable[..., object],
        neurons: int,
    ) -> None:
        if (
            self._gene_expression_data is None
            or self._gene_expression_data.shape[0] == 0
        ):
            raise ValueError("Data has not been set.")

        self._gene_expression_model = build_model(
            input_dim=self._gene_expression_data.shape[1],
            model_name="gene_expression",
            neurons=neurons,
        )

    def set_flux_model(
        self,
        build_model: Callable[..., object],
        neurons: int,
    ) -> None:
        if self._flux_data is None or self._flux_data.shape[0] == 0:
            raise ValueError("Data has not been set.")

        self._flux_model = build_model(
            input_dim=self._flux_data.shape[1],
            model_name="flux",
            neurons=neurons,
        )

    @staticmethod
    def initialise_single_view_model(
        input_dim: int,
        model_name: str,
        neurons: int,
    ) -> tf.keras.Model:
        """Initialize a model with the given parameters.

        Parameters
        ----------
        input_dim : int
            The number of input features.
        model_name : str
            The name of the model. Used for naming the layers.
        neurons : int
            The number of neurons in the hidden layers.

        Returns
        -------
        tf.keras.Model
        """
        # Input layer
        input = tf.keras.layers.Input(shape=(input_dim,))

        # Hidden layer (1)
        layer = tf.keras.layers.Dense(
            neurons,
            activation="sigmoid",
            kernel_constraint=tf.keras.constraints.max_norm(3),
            name=f"{model_name}_1",
        )(input)
        # Set 40% of input units to 0 at each update during training time
        layer = tf.keras.layers.Dropout(rate=0.4)(layer)

        # Hidden layer (2)
        layer = tf.keras.layers.Dense(
            neurons,
            activation="sigmoid",
            kernel_constraint=tf.keras.constraints.max_norm(3),
            name=f"{model_name}_2",
        )(layer)
        # Set 40% of input units to 0 at each update during training time
        layer = tf.keras.layers.Dropout(rate=0.4)(layer)

        # Final output layer
        predictions = tf.keras.layers.Dense(1, activation="linear")(layer)
        model = tf.keras.Model(inputs=input, outputs=predictions)
        return model

    def set_gene_expression_data(self, data: np.ndarray) -> None:
        """Set the gene expression data.

        Parameters
        ----------
        data : np.ndarray
            The gene expression data.

        Returns
        -------
        None
        """
        self._gene_expression_data = data

    def set_flux_data(self, data: np.ndarray) -> None:
        """Set the flux data.

        Parameters
        ----------
        data : np.ndarray
            The flux data.

        Returns
        -------
        None
        """
        self._flux_data = data
