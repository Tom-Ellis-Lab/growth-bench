import pandas as pd
from sklearn.model_selection import train_test_split


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
