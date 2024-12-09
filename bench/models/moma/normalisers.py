import abc
import dataclasses

import numpy as np
from sklearn import preprocessing as sklearn_preprocessing

from bench.models.moma.preprocessing_utils import integrators


@dataclasses.dataclass
class NormalisationParam:
    """Parameters for the normalisation.

    Attributes
    ----------
    x_train : list[integrators.OmicsData]
        The training data.
    x_val : list[integrators.OmicsData]
        The validation data.
    """

    x_train: list[integrators.OmicsData]
    x_val: list[integrators.OmicsData]


@dataclasses.dataclass
class ScaledData:
    name: str
    data: np.ndarray


@dataclasses.dataclass
class NormalisedDataset:
    scaled_x_train: list[ScaledData]
    scaled_x_val: list[ScaledData]


class Normaliser(abc.ABC):
    @abc.abstractmethod
    def normalise(self, params: NormalisationParam) -> NormalisedDataset:
        """Normalise the data.

        Parameters
        ----------
        params : NormalisationParam
            The parameters for the normalisation.

        Returns
        -------
        NormalisedDataset
            The normalised data.
        """
        pass


class StandardScalerNormaliser(Normaliser):

    def normalise(self, params: NormalisationParam) -> NormalisedDataset:
        """
        Normalise the data using StandardScaler.

        Parameters
        ----------
        params : NormalisationParam
            The parameters for the normalisation.

        Returns
        -------
        NormalisedDataset
            The normalised data.
        """

        x_train = params.x_train
        x_val = params.x_val

        scaled_x_train = []
        scaled_x_val = []

        for omics_data_train, omics_data_val in zip(x_train, x_val):
            scaler = sklearn_preprocessing.StandardScaler().fit(omics_data_train.data)
            scaled_train_data = scaler.transform(omics_data_train.data)
            scaled_val_data = scaler.transform(omics_data_val.data)

            scaled_x_train.append(
                ScaledData(name=omics_data_train.name, data=scaled_train_data)
            )
            scaled_x_val.append(
                ScaledData(name=omics_data_val.name, data=scaled_val_data)
            )

        result = NormalisedDataset(
            scaled_x_train=scaled_x_train, scaled_x_val=scaled_x_val
        )

        return result
