import keras
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import wandb


def intersect_input_data_with_growth_rates(
    input_data: pd.DataFrame, growth_rate_data: pd.DataFrame
):
    """Intersect the input data with the growth rate.
    
    Built for Ralser and Culley growth data

    Parameters
    ----------
    input_data : pd.DataFrame
        The input data.
    growth_rate_data : pd.DataFrame
        The growth rate data.

    Returns
    -------
    dict[str, pd.DataFrame]
        The intersected input data and growth rates.
        keys: input_data, growth
    """

    # INTERSECTION - intersect two dataframes
    datasets = {"input_data": input_data, "growth": growth_rate_data}
    preprocessed_data = filter_data(datasets=datasets)

    aligned_input_data = preprocessed_data["input_data"]
    aligned_growth_data = preprocessed_data["growth"]

    # remove duplicates - keep only one of the duplicates
    aligned_input_data_no_duplicates = aligned_input_data[
        ~aligned_input_data.index.duplicated(keep="first")
    ]

    # sort the dataframes by the knockout name
    aligned_input_data_no_duplicates = aligned_input_data_no_duplicates.sort_index()
    aligned_growth_data = aligned_growth_data.sort_index()

    np.random.seed(42)
    shuffled_indices = np.random.permutation(aligned_input_data_no_duplicates.index)
    shuffled_input = aligned_input_data_no_duplicates.loc[shuffled_indices]
    shuffled_growth = aligned_growth_data.loc[shuffled_indices]

    result = {
        "input_data": shuffled_input,
        "growth": shuffled_growth,
    }
    return result


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


def apply_pca_on_data(
    X_train: np.ndarray, X_test: np.ndarray, n_components: int
) -> dict[str, np.ndarray]:
    """Apply PCA to the training and test sets.

    Parameters
    ----------
    X_train : np.ndarray
        The training set features.
    X_test : np.ndarray
        The test set features.
    n_components : int
        The number of principal components to keep.

    Returns
    -------
    dict[str, np.ndarray]
        The training and test sets with PCA applied.
        keys: "train", "test", "pca"
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    result = {
        "train": X_train_pca,
        "test": X_test_pca,
        "pca": pca,
    }
    return result
