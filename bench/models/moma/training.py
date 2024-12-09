import dataclasses

from typing import Optional

import keras
import numpy as np


from bench.models.moma import normalisers
from bench.models.moma.preprocessing_utils import integrators


@dataclasses.dataclass
class LearningParams:
    """Configuration for model training.

    Attributes
    ----------
    x_train : list[np.ndarray]
        The training data.
    y_train : np.ndarray
        The training labels.
    x_val : list[np.ndarray]
        The validation data.
    y_val : np.ndarray
        The validation labels.
    epochs : int
        The number of epochs.
    batch_size : int
        The batch size.
    callbacks : Optional[list[keras.callbacks.Callback]]
        The callbacks to use during training.
    """

    x_train: list[np.ndarray]
    y_train: np.ndarray
    x_val: list[np.ndarray]
    y_val: np.ndarray
    epochs: int
    batch_size: int
    callbacks: Optional[list[keras.callbacks.Callback]] = None


def prepare_learning_params(
    x_train: list[normalisers.ScaledData],
    x_val: list[normalisers.ScaledData],
    y_train: integrators.OmicsData,
    y_val: integrators.OmicsData,
    epochs: int,
    batch_size: int,
    callbacks: Optional[list[keras.callbacks.Callback]] = None,
) -> LearningParams:
    """Prepare the learning parameters from raw data and config.

    Parameters
    ----------
    x_train : list[normalisers.ScaledData]
        The training data.
    x_val : list[normalisers.ScaledData]
        The validation data.
    y_train : integrators.OmicsData
        The training labels.
    y_val : integrators.OmicsData
        The validation labels.
    epochs : int
        The number of epochs.
    batch_size : int
        The batch size.
    callbacks : Optional[list[keras.callbacks.Callback]]
        The callbacks to use during training.

    Returns
    -------
    LearningParams
        Prepared learning parameters.
    """

    return LearningParams(
        x_train=[omics.data for omics in x_train],
        y_train=y_train.data.to_numpy(),
        x_val=[omics.data for omics in x_val],
        y_val=y_val.data.to_numpy(),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )


def train_model(model: keras.Model, params: LearningParams) -> keras.callbacks.History:
    """Train the model using the learning parameters.

    Parameters
    ----------
    model : keras.Model
        The Keras model to be trained.
    params : LearningParams
        The parameters for training the model.

    Returns
    -------
    keras.callbacks.History
        The training history.
    """
    history = model.fit(
        x=params.x_train,
        y=params.y_train,
        epochs=params.epochs,
        batch_size=params.batch_size,
        validation_data=(params.x_val, params.y_val),
        verbose=0,
        callbacks=params.callbacks,
    )
    return history
