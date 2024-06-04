from typing import Union

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
import wandb
from wandb import plot as wandb_plot


def random_split(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """Randomly split the dataset into training and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to split.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and test sets.
        keys: "train" and "test"
    """
    train_set, test_set = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    result = {"train": train_set, "test": test_set}
    return result


def apply_indices_split(
    data: pd.DataFrame, train_indices: pd.Index, test_indices: pd.Index
) -> dict[str, pd.DataFrame]:
    """Apply predefined train and test indices to split the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to split.
    train_indices : pd.Index
        The indices of the train set.
    test_indices : pd.Index
        The indices of the test set.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and test sets.
        keys: "train" and "test"
    """
    train_set = data.loc[train_indices]
    test_set = data.loc[test_indices]
    result = {"train": train_set, "test": test_set}
    return result


def plot_loss(
    loss: list[float],
    val_loss: list[float],
    plot_to_save_dir: str,
    name: str,
) -> None:
    """Plot the loss and validation loss of the model.

    Parameters
    ----------
    loss : list[float]
        The training loss.
    val_loss : list[float]
        The validation loss.
    plot_to_save_dir : str
        The directory to save the plot.
    name : str
        The name of the plot.
    """

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
    plt.savefig(plot_to_save_dir + "/" + name + ".png")
    plt.clf()

    data_train_loss = [[x, y] for (x, y) in zip(epochs, loss[20:])]
    data_val_loss = [[x, y] for (x, y) in zip(epochs, val_loss[20:])]
    table_train_loss = wandb.Table(data=data_train_loss, columns=["epochs", "loss"])
    table_val_loss = wandb.Table(data=data_val_loss, columns=["epochs", "loss"])
    wandb.log(
        {
            f"train_loss_{name}": wandb_plot.line(
                table_train_loss, "epochs", "loss", title=f"Training Loss ({name})"
            ),
            f"val_loss_{name}": wandb_plot.line(
                table_val_loss, "epochs", "loss", title=f"Validation Loss ({name})"
            ),
        }
    )


def evaluate(
    model: keras.Model,
    X_test: Union[np.ndarray, list[np.ndarray]],
    y_test: np.ndarray,
    n_outputs: int = 1,
) -> dict[str, dict[str, float]]:
    """Evaluate the model on the test set.

    Parameters
    ----------
    model : keras.Model
        The model to evaluate.
    X_test : np.ndarray
        The test set features.
    y_test : np.ndarray
        The test set labels with 3 dimensions.
    n_outputs : int
        The number of outputs of the model.

    Returns
    -------
    dict[str, dict[str, float]]
        A dictionary containing the metrics for each dimension.
    """
    results = {}

    # Evaluate the model and get predictions
    # mse, mae = model.evaluate(x=X_test, y=y_test, verbose=1)
    y_predict = model.predict(X_test)

    mae_values = []
    mse_values = []
    pearson_values = []
    spearman_values = []
    r_squared_values = []

    for i in range(n_outputs):
        y_test_i = y_test[:, i].ravel()
        y_predict_i = y_predict[:, i].ravel()

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(y_test_i - y_predict_i))
        mae_values.append(mae)

        # Mean Squared Error (MSE)
        mse = np.mean(np.square(y_test_i - y_predict_i))
        mse_values.append(mse)

        # R squared calculation
        residual = y_test_i - y_predict_i
        residual_sum_of_squares = np.sum(np.square(residual))
        total_sum_of_squares = np.sum(np.square(y_test_i - np.mean(y_test_i)))
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
        r_squared_values.append(r_squared)

        # Pearson
        pearson, _ = scipy.stats.pearsonr(y_test_i, y_predict_i)
        pearson_values.append(pearson)

        # Spearman
        spearman, _ = scipy.stats.spearmanr(y_test_i, y_predict_i)
        spearman_values.append(spearman)

        # Coverage
        non_nan_count = np.count_nonzero(~np.isnan(y_predict_i))
        total_count = len(y_predict_i)
        coverage = float(non_nan_count / total_count)

        # Plot predicted vs true values
        plt.scatter(y_test_i, y_predict_i)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Predicted vs True values for output {i+1}")
        plt.savefig(f"data/models/moma/proteomics_model_predictions_output_{i+1}.png")
        plt.clf()

        # Log the results to W&B
        columns = ["True values", "Predicted values"]
        test_table = wandb.Table(columns=columns)
        for j in range(len(y_test_i)):
            test_table.add_data(y_test_i[j], y_predict_i[j])
        wandb.log({f"proteomics_model_predictions_output_{i+1}": test_table})

        results[f"output_{i+1}"] = {
            "mae": mae,
            "mse": mse,
            "pearson": pearson,
            "spearman": spearman,
            "coverage": coverage,
            "r_squared": r_squared,
        }

    return results
