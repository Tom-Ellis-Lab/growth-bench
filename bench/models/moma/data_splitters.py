import abc

import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class DataSplitterInterface(abc.ABC):
    @abc.abstractmethod
    def split(
        self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> dict[str, pd.DataFrame]:
        """
        Split the data into two parts.

        Parameters
        ----------
        data : pd.DataFrame
            The data to split.
        test_size : float
            The fraction of the data to use as test data
        random_state : int
            The random state for reproducibility.

        Returns
        -------
        dict[str, pd.DataFrame]
            The training and test data.
            keys: "train" and "test"
        """
        pass

    @abc.abstractmethod
    def split_multiple_data(
        self,
        data: dict[str, pd.DataFrame],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Split multiple datasets in a consistent manner.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The data to split.
        test_size : float
            The fraction of the data to use as test data.
        random_state : int
            The random state for reproducibility.

        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            The training and test data.
            keys: "train" and "test"
        """
        pass

    @abc.abstractmethod
    def split_for_cross_validation(
        self,
        data: pd.DataFrame,
        cross_validation: int,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> dict[int, dict[str, pd.DataFrame]]:
        """Perform cross-validation on a single dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to split.
        cross_validation : int
            The number of cross-validation folds.
        random_state : int
            The random state for reproducibility.
        shuffle : bool
            Whether to shuffle the data before splitting.

        Returns
        -------
        dict[int, dict[str, pd.DataFrame]]
            The training and test splits for each fold.
            Keys are fold numbers, values are dictionaries with "train" and "test" DataFrames.
        """
        pass

    @abc.abstractmethod
    def split_multiple_data_for_cross_validation(
        self,
        data: dict[str, pd.DataFrame],
        cross_validation: int,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> dict[int, dict[str, dict[str, pd.DataFrame]]]:
        """Split multiple datasets for cross-validation.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The data to split.
        cross_validation : int
            The number of cross-validation folds.
        random_state : int
            The random state for reproducibility.
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        dict[int, dict[str, dict[str, pd.DataFrame]]]
            The training and test data.
            keys: "train" and "test"
        """
        pass


class DataSplitter(DataSplitterInterface):

    def split(
        self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> dict[str, pd.DataFrame]:
        """
        Split the data into two parts.

        Parameters
        ----------
        data : pd.DataFrame
            The data to split.
        test_size : float
            The fraction of the data to use as test data.
        random_state : int
            The random state for reproducibility.

        Returns
        -------
        dict[str, pd.DataFrame]
            The training and test data.
            keys: "train" and "test"
        """
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        results = {}
        results["train"] = train_set
        results["test"] = test_set
        return results

    def split_multiple_data(
        self,
        data: dict[str, pd.DataFrame],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Split multiple datasets in a consistent manner.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The data to split.
        test_size : float
            The fraction of the data to use as test data.
        random_state : int
            The random state for reproducibility.

        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            The training and test data.
            keys: "train" and "test"
        """
        self._verify_indices(data=data)
        # get the first dataframe
        reference_df = next(iter(data.values()))
        train_set, test_set = train_test_split(
            reference_df, test_size=test_size, random_state=random_state
        )

        results = {}

        results["train"] = {
            key: value.loc[train_set.index] for key, value in data.items()
        }
        results["test"] = {
            key: value.loc[test_set.index] for key, value in data.items()
        }

        return results

    def split_for_cross_validation(
        self,
        data: pd.DataFrame,
        cross_validation: int,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> dict[int, dict[str, pd.DataFrame]]:
        """Perform cross-validation on a single dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to split.
        cross_validation : int
            The number of cross-validation folds.
        random_state : int
            The random state for reproducibility.
        shuffle : bool
            Whether to shuffle the data before splitting.

        Returns
        -------
        dict[int, dict[str, pd.DataFrame]]
            The training and test splits for each fold.
            Keys are fold numbers, values are dictionaries with "train" and "test" DataFrames.
        """
        kf = KFold(
            n_splits=cross_validation, shuffle=shuffle, random_state=random_state
        )

        result = {}
        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            result[fold + 1] = {"train": train_data, "test": test_data}

        return result

    def split_multiple_data_for_cross_validation(
        self,
        data: dict[str, pd.DataFrame],
        cross_validation: int,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> dict[int, dict[str, dict[str, pd.DataFrame]]]:
        """Split multiple datasets for cross-validation.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The data to split.
        cross_validation : int
            The number of cross-validation folds.
        random_state : int
            The random state for reproducibility.
        shuffle : bool
            Whether to shuffle the data.
        """
        self._verify_indices(data=data)
        reference_df = next(iter(data.values()))
        kf = KFold(
            n_splits=cross_validation, shuffle=shuffle, random_state=random_state
        )

        result = {}
        for fold, (train_index, test_index) in enumerate(kf.split(reference_df)):
            # Split data based on indices directly within this method
            train_data = {key: value.iloc[train_index] for key, value in data.items()}
            test_data = {key: value.iloc[test_index] for key, value in data.items()}

            result[fold + 1] = {"train": train_data, "test": test_data}

        return result

    def _verify_indices(self, data: dict[str, pd.DataFrame]) -> None:
        """Verify that the dataframes have the same indices.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The dataframes to verify.
        """
        reference_index = next(iter(data.values())).index
        for key, value in data.items():
            if not value.index.equals(reference_index):
                raise ValueError(
                    f"Dataframes must have the same indices. Mismatch found in dataframe with key: '{key}'."
                )
