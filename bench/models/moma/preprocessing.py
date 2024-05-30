import keras
import pandas as pd
import wandb


def filter_data(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Filter the dataframes by the intersection of the indices.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        A dictionary where keys are dataset names and values are the dataframes.

    Returns
    -------
    dict[str, pd.DataFrame]
        The filtered dataframes, retaining the same keys as the input dictionary.
    """
    if len(datasets) < 2:
        raise ValueError("At least two datasets are required to perform intersection.")

    # Find the common indices across all dataframes
    common_knockouts = set(datasets[next(iter(datasets))].index)
    for df in datasets.values():
        common_knockouts &= set(df.index)

    # Filter each dataframe by the common indices
    result = {
        name: df[df.index.isin(common_knockouts)] for name, df in datasets.items()
    }

    return result


def get_optimiser(config: wandb.Config) -> keras.optimizers.Optimizer:
    """Get the optimizer based on the configuration.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.

    Returns
    -------
    keras.optimizers.Optimizer
        The optimizer.
    """
    if config.optimizer == "adagrad":
        optimiser = keras.optimizers.Adagrad(learning_rate=config.learning_rate)
    elif config.optimizer == "sgd":
        optimiser = keras.optimizers.SGD(
            learning_rate=config.learning_rate,
            weight_decay=config.learning_rate / config.epochs,
            momentum=config.momentum,
        )
    elif config.optimizer == "adam":
        optimiser = keras.optimizers.Adam(learning_rate=config.learning_rate)
    else:
        raise ValueError("Invalid optimizer")

    return optimiser
