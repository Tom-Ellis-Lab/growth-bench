import abc
import dataclasses

from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from bench.models.moma.preprocessing_utils import integrators


@dataclasses.dataclass
class DataSplitterParams:
    """
    Parameters for the data splitter.

    Attributes
    ----------
    data : MultiomicsData
        The data to split.
    test_size : float
        The fraction of the data to use as test data.
    random_state : int
        The random state for reproducibility.
    shuffle : bool
        Whether to shuffle the data.
    cross_validation : int
        The number of cross-validation folds.
    """

    data: integrators.MultiomicsData
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True
    cross_validation: Optional[int] = None

    def __post_init__(self):
        self._verify_indices(data=self.data.to_dict())

    def _verify_indices(self, data: dict[str, pd.DataFrame]) -> None:
        """Verify that all dataframes have the same indices."""
        reference_index = next(iter(data.values())).index
        for key, value in data.items():
            if not value.index.equals(reference_index):
                raise ValueError(
                    f"Dataframes must have the same indices. Mismatch found in dataframe with key: '{key}'."
                )


class DataSplitterInterface(abc.ABC):
    @abc.abstractmethod
    def split(
        self,
        params: DataSplitterParams,
    ) -> dict[str, pd.DataFrame]:
        """
        Split the data into two parts.

        Parameters
        ----------
        params : DataSplitterParams
            The parameters for the data splitter.

        Returns
        -------
        dict[str, pd.DataFrame]
            The training and test data.
            keys: "train" and "test"
        """
        pass


class DataSplitter(DataSplitterInterface):

    def split(
        self,
        params: DataSplitterParams,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Split multiple datasets in a consistent manner.

        Parameters
        ----------
        params : DataSplitterParams
            The parameters for the data splitter.

        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            The training and test data.
            keys: "train" and "test"
        """
        data = params.data.to_dict()
        # get the first dataframe
        reference_df = next(iter(data.values()))
        train_set, test_set = train_test_split(
            reference_df, test_size=params.test_size, random_state=params.random_state
        )

        results = {}

        results["train"] = {
            key: value.loc[train_set.index] for key, value in data.items()
        }
        results["test"] = {
            key: value.loc[test_set.index] for key, value in data.items()
        }

        return results


class CrossValidationDataSplitter(DataSplitterInterface):

    def split(
        self,
        params: DataSplitterParams,
    ) -> dict[int, dict[str, dict[str, pd.DataFrame]]]:
        """Split multiple datasets for cross-validation.

        Parameters
        ----------
        params : DataSplitterParams
            The parameters for the data splitter.
        """
        if params.cross_validation is None:
            raise ValueError("Cross-validation parameter must be provided.")
        data = params.data.to_dict()
        # get the first dataframe
        reference_df = next(iter(data.values()))
        kf = KFold(
            n_splits=params.cross_validation,
            shuffle=params.shuffle,
            random_state=params.random_state,
        )

        result = {}
        for fold, (train_index, test_index) in enumerate(kf.split(reference_df)):
            # Split data based on indices directly within this method
            train_data = {key: value.iloc[train_index] for key, value in data.items()}
            test_data = {key: value.iloc[test_index] for key, value in data.items()}

            result[fold + 1] = {"train": train_data, "test": test_data}

        return result
