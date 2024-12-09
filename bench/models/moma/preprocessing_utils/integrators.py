import abc
import dataclasses
from typing import Optional

import numpy as np
import pandas as pd

from bench.models.moma.preprocessing_utils import filters


@dataclasses.dataclass
class OmicsData:
    """Dataclass for omics data.

    Attributes
    ----------
    name : str
        The name of the omics data.
    data : pd.DataFrame
        The omics data.
    """

    name: str
    data: pd.DataFrame


@dataclasses.dataclass
class ProteomicsData(OmicsData):
    """Dataclass for proteomics data.

    Attributes
    ----------
    name : str = "proteomics"
        The name of the proteomics data.
    data : pd.DataFrame
        The proteomics data.
    """

    data: pd.DataFrame


@dataclasses.dataclass
class MultiomicsData:
    """Dataclass for multiomics data.

    Attributes
    ----------
    proteomics : Optional[OmicsData]
        The proteomics data.
    transcriptomics : Optional[OmicsData]
        The transcriptomics data.
    fluxomics : Optional[OmicsData]
        The fluxomics data.
    growth : Optional[OmicsData]
        The growth data.
    """

    proteomics: Optional[OmicsData] = None
    transcriptomics: Optional[OmicsData] = None
    fluxomics: Optional[OmicsData] = None
    growth: Optional[OmicsData] = None

    def to_dict(self):
        result = {}
        for attr_name, omics_data in dataclasses.asdict(self).items():
            if omics_data is not None:
                result[omics_data["name"]] = omics_data["data"]
        return result


class DataIntegrator(abc.ABC):
    @abc.abstractmethod
    def integrate(self, data: MultiomicsData) -> dict[str, pd.DataFrame]:
        """Integrate the multiomics data.

        Parameters
        ----------
        multiomics_data : MultiomicsData
            The multiomics data.

        Returns
        -------
        dict[str, pd.DataFrame]
            The integrated data.
        """
        pass


class OmicsDataIntegrator(DataIntegrator):

    def integrate(self, multiomics_data: MultiomicsData) -> dict[str, pd.DataFrame]:
        """Integrate the multiomics data.

        Parameters
        ----------
        multiomics_data : MultiomicsData
            The multiomics data.

        Returns
        -------
        dict[str, pd.DataFrame]
            The integrated data.
        """
        data = self._intersect_dataframes(data=multiomics_data)
        data = self._remove_duplicates(data=data)
        data = self._sort(datasets=data)
        result = self._shuffle(datasets=data)
        return result

    def _intersect_dataframes(
        self,
        data: MultiomicsData,
    ) -> dict[str, pd.DataFrame]:
        """Intersect the transcriptomics, fluxomics, and growth data.

        Parameters
        ----------
        data : MultiomicsData
            The data to intersect.

        Returns
        -------
        dict[str, pd.DataFrame]
            The intersected data.
        """
        datasets = data.to_dict()

        factory = filters.FilterFactory()
        filter = factory.create_filter(filter_type=filters.FilterTypes.INTERSECTION)
        result = filter.filter_data(data=datasets)

        return result

    def _remove_duplicates(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Remove duplicates from the dataframes.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            The dataframes to remove duplicates from.

        Returns
        -------
        dict[str, pd.DataFrame]
            The dataframes without duplicates.
        """
        datasets = {
            key: df[~df.index.duplicated(keep="first")] for key, df in data.items()
        }
        return datasets

    def _sort(self, datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Sort the dataframes by the knockout name.

        Parameters
        ----------
        datasets : dict[str, pd.DataFrame]
            The dataframes to sort.

        Returns
        -------
        dict[str, pd.DataFrame]
            The sorted dataframes.
        """
        sorted_datasets = {key: df.sort_index() for key, df in datasets.items()}
        return sorted_datasets

    def _shuffle(
        self,
        datasets: dict[str, pd.DataFrame],
        random_state: int = 42,
    ) -> dict[str, pd.DataFrame]:
        """Shuffle the dataframes.

        Parameters
        ----------
        datasets : dict[str, pd.DataFrame]
            The dataframes to shuffle.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        dict[str, pd.DataFrame]
            The shuffled dataframes.
        """
        np.random.seed(random_state)

        shuffled_datasets = {}

        # Get shuffled indices based on the index of the first DataFrame
        first_key = next(iter(datasets))
        shuffled_indices = np.random.permutation(datasets[first_key].index)

        # Shuffle each DataFrame in the dictionary
        for key, df in datasets.items():
            shuffled_datasets[key] = df.loc[shuffled_indices]

        return shuffled_datasets
