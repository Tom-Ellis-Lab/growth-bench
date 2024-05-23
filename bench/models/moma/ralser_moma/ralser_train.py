import pandas as pd


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
