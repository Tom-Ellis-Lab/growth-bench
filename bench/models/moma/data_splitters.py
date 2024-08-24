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


@dataclasses.dataclass
class LearningData:
    """
    Data for learning.

    Attributes
    ----------
    x_train : list[integrators.OmicsData]
        The training data.
    y_train : integrators.OmicsData
        The training labels.
    x_val : list[integrators.OmicsData]
        The validation data.
    y_val : integrators.OmicsData
        The validation labels.
    """

    x_train: list[integrators.OmicsData]
    y_train: integrators.OmicsData
    x_val: list[integrators.OmicsData]
    y_val: integrators.OmicsData


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
    ) -> LearningData:
        """Split multiple datasets in a consistent manner.

        Parameters
        ----------
        params : DataSplitterParams
            The parameters for the data splitter.

        Returns
        -------
        LearningData
            The training and validation data.
        """
        data = params.data.to_dict()
        # get the first dataframe
        reference_df = next(iter(data.values()))
        train_indices, test_indices = train_test_split(
            reference_df.index,
            test_size=params.test_size,
            random_state=params.random_state,
        )

        # Training set
        x_train = [
            integrators.OmicsData(name=key, data=df.loc[train_indices])
            for key, df in data.items()
            if key != "growth"
        ]

        # Validation set
        x_val = [
            integrators.OmicsData(name=key, data=df.loc[test_indices])
            for key, df in data.items()
            if key != "growth"
        ]

        # Training target set
        y_train = integrators.OmicsData(
            name="growth",
            data=data["growth"].loc[train_indices],
        )

        # Validation target set
        y_val = integrators.OmicsData(
            name="growth",
            data=data["growth"].loc[test_indices],
        )

        results = LearningData(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )

        return results


@dataclasses.dataclass
class CrossValidationLearningData:
    """
    Data for learning with cross-validation.
    Each attribute is a list of data for each fold.

    Attributes
    ----------
    x_train : list[list[integrators.OmicsData]]
        The training data.
    y_train : list[integrators.OmicsData]
        The training labels.
    x_val : list[list[integrators.OmicsData]]
        The validation data.
    y_val : list[integrators.OmicsData]
        The validation labels.
    """

    x_train: list[list[integrators.OmicsData]]
    y_train: list[integrators.OmicsData]
    x_val: list[list[integrators.OmicsData]]
    y_val: list[integrators.OmicsData]


class CrossValidationDataSplitter(DataSplitterInterface):

    def split(
        self,
        params: DataSplitterParams,
    ) -> CrossValidationLearningData:
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

        x_train = []
        y_train = []
        x_val = []
        y_val = []

        for train_index, test_index in kf.split(reference_df):
            # Training set
            x_train_fold = [
                integrators.OmicsData(name=key, data=df.iloc[train_index])
                for key, df in data.items()
                if key != "growth"
            ]

            # Validation set
            x_val_fold = [
                integrators.OmicsData(name=key, data=df.iloc[test_index])
                for key, df in data.items()
                if key != "growth"
            ]

            # Training target set
            y_train_fold = integrators.OmicsData(
                name="growth",
                data=data["growth"].iloc[train_index],
            )

            # Validation target set
            y_val_fold = integrators.OmicsData(
                name="growth",
                data=data["growth"].iloc[test_index],
            )

            x_train.append(x_train_fold)
            y_train.append(y_train_fold)
            x_val.append(x_val_fold)
            y_val.append(y_val_fold)

        result = CrossValidationLearningData(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )
        return result
