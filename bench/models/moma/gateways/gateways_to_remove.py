import abc
import pathlib

from typing import Any

import pandas as pd
import pyreadr


class Gateway(abc.ABC):
    @abc.abstractmethod
    def get(self) -> Any:
        """
        Get the data.

        Returns
        -------
        Any
            The data.
        """
        pass


class ProteomicsGateway(Gateway):

    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path

    def get(self) -> pd.DataFrame:
        """
        Get the proteomics data.

        Returns
        -------
        pd.DataFrame
            The proteomics data.
        """
        return pd.read_csv(self.file_path)


class Yeast5kGrowthRatesGateway(Gateway):

    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path

    def get(self) -> pd.DataFrame:
        """
        Get the yeast5k growth rates data.

        Returns
        -------
        pd.DataFrame
            The yeast5k growth rates
        """
        return pd.read_csv(self.file_path)


class CulleyDataGateway(Gateway):

    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path

    def get(self) -> pd.DataFrame:
        """Get the culley data.

        TODO: Explain what culley data is
        Returns
        -------
        pd.DataFrame
            The culley dat.
        """
        result = pyreadr.read_r(self.file_path)[None]
        return result


class TranscriptomicsGateway(Gateway):

    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path

    def get(self) -> pd.DataFrame:
        """
        TODO: Add docstring
        """
        transcritpomics_data = pyreadr.read_r(self.file_path)[None]
        return transcritpomics_data


class FluxomicsGateway(Gateway):

    def __init__(
        self,
        transcriptomicsGateway: TranscriptomicsGateway,
        culley_gateway: CulleyDataGateway,
    ):
        self.transcriptomics_gateway = transcriptomicsGateway
        self.culley_gateway = culley_gateway

    def get(self) -> pd.DataFrame:
        """
        TODO: Add docstring
        """
        transcriptomics_data = self.transcriptomics_gateway.get()
        culley_data = self.culley_gateway.get()
        result = culley_data.drop(columns=transcriptomics_data.columns.values)
        return result


class DuibhirGrowthRatesGateway(Gateway):

    def __init__(
        self,
        culley_gateway: CulleyDataGateway,
    ):
        self.culley_gateway = culley_gateway

    def get(self) -> pd.DataFrame:
        """
        TODO: Add docstring
        """
        culley_data = self.culley_gateway.get()
        result = culley_data[["Row", "log2relT"]]
        return result


class GeneNamesGateway(Gateway):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path, sep="\t")

    def get_systematic_name(self) -> str:
        return "systematic_name"

    def get_standard_name(self) -> str:
        return "standard_name"
