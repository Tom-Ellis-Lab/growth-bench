from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf


def split(data: pd.DataFrame, test_indices: list[int]) -> dict[str, pd.DataFrame]:
    """Split the dataset into training and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to split.
    test_indices : list[int]
        The indices of the test set.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and test sets.
        keys: "train" and "test"
    """
    test_set = data.iloc[test_indices, :]
    train_set = data.drop(data.index[test_indices])
    result = {"train": train_set, "test": test_set}

    return result


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    weights_to_save_dir: str,
    weights_name: str,
    callbacks: list[tf.keras.callbacks.Callback] = [],
) -> tf.keras.callbacks.History:
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["mean_absolute_error"],
    )

    result = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=True,
        callbacks=callbacks,
    )
    model.save_weights(
        weights_to_save_dir + "proteomics" + weights_name + ".weights.h5"
    )
    model.save(weights_to_save_dir + "proteomics" + weights_name + ".model.keras")
    return result


def _plot_loss(loss: list[float], val_loss: list[float], plot_to_save_dir: str) -> None:
    """Plot the loss and validation loss of the model.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        The history of the model training.
    """

    # epochs = range(1, len(history.history["loss"]) + 1)
    # plt.plot(epochs, history.history["loss"], "bo", label="Training loss")
    # plt.plot(epochs, history.history["val_loss"], "b", label="Validation loss")

    # Generate the epoch numbers starting from 21 since you cut off the first 20
    epochs = range(21, len(loss) + 1)
    # Plot the loss, skipping the first 20 epochs
    plt.plot(epochs, loss[20:], label="train")
    plt.plot(epochs, val_loss[20:], label="validation")

    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # Save the plot
    plt.savefig(plot_to_save_dir + "/proteomics_model_loss.png")


def _evaluate(
    model: tf.keras.Model,
    X_test: Union[np.ndarray, list[np.ndarray]],
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the model on the test set.

    Parameters
    ----------
    model : tf.keras.Model
        The model to evaluate.
    X_test : np.ndarray
        The test set features.
    y_test : np.ndarray
        The test set labels.

    Returns
    -------
    float
        The mean absolute error of the model on the test set.
    """
    mse, mae = model.evaluate(x=X_test, y=y_test, verbose=1)
    y_predict = model.predict(X_test)
    y_predict = y_predict.ravel()
    y_test = y_test.ravel()

    # R squared calculation
    residual = y_test - y_predict
    residual_sum_of_squares = np.sum(np.square(residual))
    total_sum_of_squares = np.sum(np.square(y_test - np.mean(y_test)))
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    # Pearson
    pearson, _ = scipy.stats.pearsonr(y_test, y_predict)
    # Spearman
    spearman, _ = scipy.stats.spearmanr(y_test, y_predict)

    # Coverage
    non_nan_count = np.count_nonzero(~np.isnan(y_predict))
    total_count = len(y_predict)
    coverage = float(non_nan_count / total_count)

    result = {
        "mae": mae,
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        "coverage": coverage,
        "r_squared": r_squared,
    }
    return result
