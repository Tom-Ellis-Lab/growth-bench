import pandas as pd


import wandb
from wandb import plot as wandb_plot


def plot_model_loss(
    data: dict[str, dict[str, pd.DataFrame]],
    history: dict[str, list[float]],
    config: wandb.Config,
    fold: int | None = None,
) -> None:
    """Plot the model loss.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        The data used for training.
    history : dict[str, list[float]]
        The history of the model.
    config : wandb.Config
        The configuration object.
    fold : int, optional
        The fold number, by default None.
    """
    normalized_loss = get_normalised_loss(
        data=data,
        history=history,
        config=config,
        type_of_loss="loss",
    )

    normalized_val_loss = get_normalised_loss(
        data=data,
        history=history,
        config=config,
        type_of_loss="val_loss",
    )
    if fold:
        name = f"fold_{fold+1}"
    elif len(config.input_type) == 1:
        name = config.input_type[0]
    else:
        name = "multimodal"

    plot_loss_curves(
        loss=normalized_loss,
        val_loss=normalized_val_loss,
        name=f"{name}",
    )


def get_normalised_loss(
    data: dict[str, dict[str, pd.DataFrame]],
    history: dict[str, list[float]],
    config: wandb.Config,
    type_of_loss: str,
) -> list[float]:
    """Get the normalised loss.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        The data used for training.

    history : dict[str, list[float]]
        The history of the model.

    config : wandb.Config
        The configuration object.

    type_of_loss : str
        The type of loss to normalise.

    Returns
    -------
    list[float]
        The normalised loss.
    """
    if type_of_loss == "loss":
        total_samples = get_total_samples(data=data["train"])
        loss = history["loss"]
        normalised_loss = [
            loss_value * config.batch_size / total_samples for loss_value in loss
        ]
    elif type_of_loss == "val_loss":
        total_samples = get_total_samples(data=data["test"])
        val_loss = history["val_loss"]
        normalised_loss = [
            val_loss_value * config.batch_size / total_samples
            for val_loss_value in val_loss
        ]
    else:
        raise ValueError("The type of loss is not recognised.")
    return normalised_loss


def get_total_samples(
    data: dict[str, pd.DataFrame],
) -> int:
    """Get the total number of training samples.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        The training data.

    Returns
    -------
    int
        The total number of training samples.
    """
    for key, value in data.items():
        if key != "growth":
            total_samples = len(value)
            return total_samples
    raise ValueError("The training data does not contain any samples.")


def plot_loss_curves(
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


def log_configuations(config: wandb.Config) -> None:
    """Log the configurations to wandb.

    Parameters
    ----------

    config : wandb.Config
        The configuration object.

    """
    print("Configuration Table")
    print("-" * 60)
    print(f"{'Hyperparameter':<15}{'Value':<15}")
    print("-" * 60)
    print(f"{'epochs':<15}{config.epochs:<15}")
    print(
        f"{'neurons':<15}{', '.join(f'{k}: {v}' for k, v in config.neurons.items()):<15}"
    )
    print(f"{'batch size':<15}{config.batch_size:<15}")
    print(f"{'learning rate':<15}{config.learning_rate:<15}")
    print(f"{'optimizer':<15}{config.optimizer:<15}")
    print(f"{'momentum':<15}{config.momentum:<15}")
    print(
        f"{'dropout':<15}{', '.join(f'{k}: {v}' for k, v in config.dropout.items()):<15}"
    )
    print(f"{'medium':<15}{config.medium:<15}")
    print(f"{'modalities':<15}{' '.join(str(i) for i in config.input_type):<15}")
    print(f"{'set seed':<15}{str(bool(config.set_seed)):<15}")
    print("-" * 60)


def log_results(
    validation_results: dict[str, dict[str, float]], config: wandb.Config
) -> None:
    """Log the results to wandb.

    TODO: Modify docstring, do not pass wandb.Config as parameter

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        The results to log.
    """
    print("=====================================")
    print("TRAINING AND VALIDATION DONE")
    log_configuations(config=config)
    print("=====================================")
    print("Results (to copy)")
    print(validation_results)
    print("=====================================")
    print("Results")
    result = {}
    for fold, single_run_results in validation_results.items():
        print(f"Fold {fold}: {single_run_results}")
        result[str(fold)] = single_run_results  # wandb accept only string keys
    wandb.log(result)


def log_single_fold_results(result: dict[str, float], fold: int) -> None:
    """Log the results of a single fold.

    Parameters
    ----------
    result : dict[str, float]
        The results of the fold.
    fold : int
        The fold number.
    """
    for key, value in result.items():
        print(f"FOLD {fold+1} - {key}: {value}")
    print(f"DONE: Training on fold {fold+1}/5 completed")
