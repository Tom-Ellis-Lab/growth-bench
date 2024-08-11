import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
import wandb
from wandb import plot as wandb_plot
from wandb.integration.keras import WandbMetricsLogger


from bench.models.moma import preprocessing, train


def train_and_evaluate(
    config: wandb.Config,
    model: keras.Model,
    data: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, dict[str, float]]:
    """Train the model, and evaluate it.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.

    data : dict[str, pd.DataFrame]
        The data to use for training and testing.

    Returns
    -------
    dict[str, dict[str, float]]
        The results of the model evaluation.
    """

    optimiser = preprocessing.get_optimiser(config=config)

    model.compile(
        optimizer=optimiser,
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )

    y_train = data["train"]["growth"]
    y_train = y_train.to_numpy()
    y_test = data["test"]["growth"]
    y_test = y_test.to_numpy()

    history = model.fit(
        x=[
            data["scaled_train"][key] for key in data["train"].keys() if key != "growth"
        ],
        y=y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(
            [
                data["scaled_test"][key]
                for key in data["test"].keys()
                if key != "growth"
            ],
            data["test"]["growth"],
        ),
        verbose=0,
        callbacks=[WandbMetricsLogger()],
    )

    if config.save_weights:
        print("\n==== SAVING WEIGHTS ====\n")

        name = f"{config.input_type}_medium{config.medium}_lr{config.learning_rate}_epochs{config.epochs}_batch{config.batch_size}_neurons{config.neurons}_optimiser{config.optimizer}"

        model.save_weights(f"data/models/moma/{name}.weights.h5")

    X_test = [
        data["scaled_test"][key] for key in data["test"].keys() if key != "growth"
    ]

    results = train.evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )
    results[config.medium] = results.pop("output_1")
    results[config.medium]["history"] = history.history
    return results


def random_split(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """TODO: Deprecated - Randomly split the dataset into training and test sets.

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
    """TODO: Deprecated - Apply predefined train and test indices to split the dataset.

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


def evaluate(
    model: keras.Model,
    X_test: list[np.ndarray],
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
