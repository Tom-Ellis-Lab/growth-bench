import abc
import dataclasses

import pandas as pd
from sklearn import preprocessing as sklearn_preprocessing


@dataclasses.dataclass
class NormalisationParam:
    """Parameters for the normalisation.

    Attributes
    ----------
    data : dict[str, dict[str, pd.DataFrame]]
        The data to normalise.
    target_name : str
        The name of the target column
    """

    data: dict[str, dict[str, pd.DataFrame]]
    target_name: str = "target"

    def __post_init__(self):
        self._verify(data=self.data, target_name=self.target_name)

    def _verify(
        self, data: dict[str, dict[str, pd.DataFrame]], target_name: str
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Verify that:
        - the dict has only two keys: "train" and "test",
        - the values are dict of keys and dataframes.
        - the dataframes have the target column.

        Parameters
        ----------
        data : dict[str, dict[str, pd.DataFrame]]
            The data to verify.
        target_name : str
            The name of the target column

        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            The verified data.
        """
        if len(data) != 2:
            raise ValueError("Data must have exactly two keys: 'train' and 'test'.")

        for key, value in data.items():
            if key not in ["train", "test"]:
                raise ValueError(
                    f"Unknown key: {key}. Only 'train' and 'test' are allowed."
                )

            if not isinstance(value, dict):
                raise ValueError("The values must be dictionaries.")

            has_target = False
            for inner_key, inner_value in value.items():
                if inner_key == target_name:
                    has_target = True
                if not isinstance(inner_key, str):
                    raise ValueError("The keys must be strings.")

                if not isinstance(inner_value, pd.DataFrame):
                    raise ValueError("The values must be pandas DataFrames.")

            if not has_target:
                raise ValueError(f"Data must contain the target: {target_name}.")

        return data


class Normaliser(abc.ABC):
    @abc.abstractmethod
    def normalise(
        self, params: NormalisationParam
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Normalise the data.

        Parameters
        ----------
        params : NormalisationParam
            The parameters for the normalisation.
        """
        pass


class StandardScalerNormaliser(Normaliser):

    def normalise(
        self, params: NormalisationParam
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Normalise the data using StandardScaler.

        Parameters
        ----------
        params : NormalisationParam
            The parameters for the normalisation.

        Returns
        -------
        dict[str, dict[str, pd.DataFrame]]
            The original data with the scaled data added (keys: "scaled_train", "scaled_test").
        """
        data = params.data
        train_data = data["train"]
        test_data = data["test"]

        scaled_train = {}
        scaled_test = {}

        for key, value in train_data.items():
            if key != params.target_name:
                scaler = sklearn_preprocessing.StandardScaler().fit(value)
                scaled_train[key] = scaler.transform(value)
                scaled_test[key] = scaler.transform(test_data[key])

        data["scaled_train"] = scaled_train
        data["scaled_test"] = scaled_test
        return data
