import keras
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as sklearn_preprocessing
import wandb

from bench.models.moma.ralser_moma import ralser_preprocessing
from bench.models.moma.culley_moma import culley_main

from sklearn.model_selection import KFold


def get_data(medium: str, input_type: list[str]) -> dict[str, pd.DataFrame]:
    """Get the data based on the medium and input type.

    Parameters
    ----------
    medium : str
        The medium used for the growth data.

    input_type : list[str]
        The omics data to include.
    """
    growth_rate_medium = {medium: f"growth_rate_{medium}"}

    culley_data_loaded = False
    ralser_data_loaded = False

    if medium in ["SC", "SM", "YPD"]:
        ralser_data = ralser_preprocessing.get_ralser_data(
            cols_growth_data=growth_rate_medium
        )
        growth_data = ralser_data["growth"]
        ralser_data_loaded = True
    if medium == "SD":
        culley_data = culley_main.get_culley_data()
        growth_data = culley_data["growth"]
        culley_data_loaded = True

    input_data = {}
    for omics in input_type:
        if omics == "proteomics":
            if ralser_data_loaded:
                input_data[omics] = ralser_data["proteomics"]
            else:
                ralser_data = ralser_preprocessing.get_ralser_data(
                    cols_growth_data=growth_rate_medium
                )
            input_data[omics] = ralser_data["proteomics"]
        elif omics in ["transcriptomics", "fluxomics"]:
            if culley_data_loaded:
                input_data[omics] = culley_data[omics]
            else:
                culley_data = culley_main.get_culley_data()
            input_data[omics] = culley_data[omics]

    results = input_data.copy()
    results["growth"] = growth_data
    return results


def scale_data(
    data: dict[str, dict[str, pd.DataFrame]]
) -> dict[str, dict[str, pd.DataFrame]]:
    """Scale the data using StandardScaler.

    Parameters
    ----------
    data : dict[str, dict[str, pd.DataFrame]]
        The data to scale.

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        The scaled data.
    """
    train_data = data["train"]
    test_data = data["test"]

    scaled_train = {}
    scaled_test = {}

    for key, value in train_data.items():
        if key != "growth":
            scaler = sklearn_preprocessing.StandardScaler().fit(value)
            scaled_train[key] = scaler.transform(value)
            scaled_test[key] = scaler.transform(test_data[key])

    data["scaled_train"] = scaled_train
    data["scaled_test"] = scaled_test
    return data


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


def split_data(
    data: dict[str, pd.DataFrame],
    set_seed: bool,
    cross_validation: int | None,
) -> dict[int, dict[str, dict[str, pd.DataFrame]]]:
    """
    TODO: Add docstring
    """
    if cross_validation is None:
        # No cross-validation, simple holdout split
        result = {}
        split_data = split_data_using_names(data=data, set_seed=set_seed)
        result[0] = split_data
    else:
        # Cross-validation
        result = {}
        kf = KFold(n_splits=cross_validation, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(data["growth"])):
            result[fold + 1] = split_data_using_indices(
                data=data, train_index=train_index, test_index=test_index
            )

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
