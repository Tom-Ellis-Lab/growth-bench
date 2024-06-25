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


def split_data_using_indices(
    data: dict[str, pd.DataFrame], train_index: np.ndarray, test_index: np.ndarray
) -> dict[str, dict[str, pd.DataFrame]]:
    """Split the data based on the indices.

    Parameters
    ----------
    data : pd.DataFrame
        The data to split.

    train_index : np.ndarray
        The indices for the training data.

    test_index : np.ndarray
        The indices for the test data.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and test data.
    """

    train_indices = {key: value.index[train_index] for key, value in data.items()}
    test_indices = {key: value.index[test_index] for key, value in data.items()}

    train_data = {key: value.loc[train_indices[key]] for key, value in data.items()}
    test_data = {key: value.loc[test_indices[key]] for key, value in data.items()}

    result = {"train": train_data, "test": test_data}
    return result


def split_data_using_names(
    data: dict[str, pd.DataFrame], set_seed: bool
) -> dict[str, dict[str, pd.DataFrame]]:
    """Split the data based on the names.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        The data to split.

    set_seed : bool
        Whether to set the seed for reproducibility.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and test sets.
        keys: "train" and "test"
    """

    if set_seed:
        train_set, test_set = train_test_split(
            data["growth"], test_size=0.2, random_state=42
        )
    else:
        train_set, test_set = train_test_split(data["growth"], test_size=0.2)

    train_set_indices = train_set.index
    test_set_indices = test_set.index

    results = {}
    results["train"] = {
        key: value.loc[train_set_indices] for key, value in data.items()
    }
    results["test"] = {key: value.loc[test_set_indices] for key, value in data.items()}
    return results


def compute_error_on_selected_genes(
    selected_genes: list[str],
    data: dict[str, dict[str, pd.DataFrame]],
    trained_model: keras.Model,
    input_type: list[str],
    medium: str,
) -> dict[str, dict[str, float]]:
    """Compute the error on the selected genes.

    Parameters
    ----------
    selected_genes : list[str]
        The selected genes for analysis.

    data : dict[str, pd.DataFrame]
        The data used for training.

    trained_model : keras.Model
        The trained model.

    input_type : list[str]
        The input type.

    medium : str
        The medium to use.
    """
    selected_genes_indices = [
        data["test"][input_type[0]].index.get_loc(gene) for gene in selected_genes
    ]
    y_test = data["test"]["growth"]
    selected_y_test = y_test.loc[selected_genes]
    X_test = [
        data["scaled_test"][key] for key in data["test"].keys() if key != "growth"
    ]
    selected_X_test = [X[np.ix_(selected_genes_indices)] for X in X_test]
    selected_y_pred = trained_model.predict(selected_X_test)

    individual_mae = np.abs(selected_y_pred - selected_y_test)
    individual_mse = (selected_y_pred - selected_y_test) ** 2

    print("=====================================")
    print("ANALYSIS ON SELECTED GENES")
    print("Individual MAE:", individual_mae)
    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    print("Individual MSE:", individual_mse)
    print("=====================================")
    result = {}
    for i, gene in enumerate(selected_genes):
        result[gene] = {
            "MAE": individual_mae[f"growth_rate_{medium}"].iloc[i],
            "MSE": individual_mse[f"growth_rate_{medium}"].iloc[i],
        }

    return result


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


def plot_loss(
    loss: list[float],
    val_loss: list[float],
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


def plot_lossxx(
    loss: list[float],
    val_loss: list[float],
    plot_to_save_dir: str,
    name: str,
) -> None:
    """TODO: Deprecated - Plot the loss and validation loss of the model.

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
