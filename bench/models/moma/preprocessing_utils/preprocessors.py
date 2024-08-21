import abc
import enum

import pandas as pd

from bench.models.moma.preprocessing_utils import mappers
from bench.models.moma.gateways import gateways


class GrowthMedium(enum.Enum):
    SC = "SC"
    SM = "SM"
    YPD = "YPD"
    SD = "SD"


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess(self) -> pd.DataFrame:
        pass


class ProteomicsPreprocessor(Preprocessor):

    def __init__(self, gateway: gateways.ProteomicsGateway):
        self.gateway = gateway

    def preprocess(self) -> pd.DataFrame:
        """Preprocess the proteomics data.

        Returns
        -------
        pd.DataFrame
            The preprocessed proteomics data.
        """
        data = self.gateway.get()
        proteomics_data = self._remove_quality_controls(data=data)
        proteomics_data = self._remove_his3delta_controls(data=proteomics_data)
        proteomics_data = self._rename_columns(data=proteomics_data)
        proteomics_data = self._set_index(data=proteomics_data)
        proteomics_data = self._transpose_data(data=proteomics_data)

        return proteomics_data

    def _remove_quality_controls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove quality control columns from the dataset (_qc_qc_qc).

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to remove quality control columns from.

        Returns
        -------
        pd.DataFrame
            The dataset without quality control columns.
        """
        # Remove the quality control columns (they contain "_qc_qc_qc")
        quality_control_data = data.filter(regex="_qc_qc_qc", axis=1)
        result = data.drop(quality_control_data.columns, axis=1)
        return result

    def _remove_his3delta_controls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove the his3Δ control strains from the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to remove the his3Δ control strains from.

        Returns
        -------
        pd.DataFrame
            The dataset without the his3Δ control strains.
        """
        his3_control_data = data.filter(regex="_HIS3", axis=1)
        result = data.drop(his3_control_data.columns, axis=1)
        return result

    def _rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Rename the columns in the dataset to the systematic gene names.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to rename the columns for.

        Returns
        -------
        pd.DataFrame
            The dataset with renamed columns.
        """
        new_columns = [
            (
                data.columns[i]
                if i < 1
                else self._rename_single_column(col=data.columns[i])
            )
            for i in range(len(data.columns))
        ]
        data.columns = new_columns
        return data

    def _rename_single_column(self, col: str) -> str:
        """Rename the original column in dataset to the systematic gene name.

        TODO: Visualise how the original column looks like

        Parameters
        ----------
        col : str
            The original column name.

        Returns
        -------
        str
            The systematic gene name.
        """
        if "_ko_" in col:
            parts = col.split("_ko_")
            # Further split by "_" and take the first element which is the systematic gene name
            gene_name = parts[1].split("_")[0]
            return gene_name
        return col

    def _set_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Set the first column as the index of the dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe to set the index for.

        Returns
        -------
        pd.DataFrame
            The dataframe with the index set.
        """
        data.set_index(data.columns[0], inplace=True)
        data.index.name = "protein_id"
        return data

    def _transpose_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transpose the data so that the rows are knockouts and the columns are proteins.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe to transpose.

        Returns
        -------
        pd.DataFrame
            The transposed dataframe.
        """
        return data.T


class Yeast5kGrowthRatesPreprocessor(Preprocessor):

    def __init__(self, gateway: gateways.Yeast5kGrowthRatesGateway):
        self.gateway = gateway

    def preprocess(self, growth_medium: GrowthMedium) -> pd.DataFrame:
        """Preprocess the growth rates data.

        Parameters
        ----------
        growth_medium : GrowthMedium
            The growth medium to preprocess the data for.

        Returns
        -------
        pd.DataFrame
            The preprocessed growth rates data.
        """
        data = self.gateway.get()
        data = data[["orf", growth_medium.value]]

        data.rename(columns={"orf": "systematic_name"}, inplace=True)

        data.rename(
            columns={growth_medium.value: f"growth_rate_{growth_medium.value}"},
            inplace=True,
        )
        data.set_index("systematic_name", inplace=True)
        return data


class TranscriptomicsPreprocessor(Preprocessor):

    _knockout_id = "knockout_id"

    def __init__(
        self,
        transcriptomics_gateway: gateways.TranscriptomicsGateway,
        culley_gateway: gateways.CulleyDataGateway,
        gene_names_mapper: mappers.Mapper,
    ):
        self.transcriptomics_gateway = transcriptomics_gateway
        self.culley_gateway = culley_gateway
        self.gene_names_mapper = gene_names_mapper

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the transcriptomics data.

        Returns
        -------
        pd.DataFrame
            The preprocessed transcriptomics data.
        """
        data = self.transcriptomics_gateway.get()
        data = self._add_knockout_id(data)

        data = self.gene_names_mapper.map(
            data=data,
            target_col=self._knockout_id,
            new_col=self._knockout_id,
            fillna_col=self._knockout_id,
        )

        data = self._remove_zero_columns(data=data)
        data.drop(columns=["log2relT"], inplace=True)
        data.set_index(self._knockout_id, inplace=True)
        return data

    def _add_knockout_id(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add knockout id column to the data.
        """
        data[self._knockout_id] = self.culley_gateway.get()["Row"]
        return data

    def _remove_zero_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with only zeros.

        Parameters
        ----------
        data : pd.DataFrame
            The data to remove columns from.

        Returns
        -------
        pd.DataFrame
            The data with zero columns removed.
        """
        return data.loc[:, (data != 0).any(axis=0)]


class FluxomicsPreprocessor(Preprocessor):

    _knockout_id = "knockout_id"

    def __init__(
        self,
        fluxomics_gateway: gateways.FluxomicsGateway,
        gene_names_mapper: mappers.Mapper,
    ):
        self.fluxomics_gateway = fluxomics_gateway
        self.gene_names_mapper = gene_names_mapper

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the fluxomics data.

        Returns
        -------
        pd.DataFrame
            The preprocessed fluxomics data.
        """

        data = self.fluxomics_gateway.get()

        data = self._remove_zero_columns(data=data)
        data.rename(columns={"Row": self._knockout_id}, inplace=True)

        data = self.gene_names_mapper.map(
            data=data,
            target_col=self._knockout_id,
            new_col=self._knockout_id,
            fillna_col=self._knockout_id,
        )

        data.set_index(self._knockout_id, inplace=True)
        return data

    def _remove_zero_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with only zeros.

        Parameters
        ----------
        data : pd.DataFrame
            The data to remove columns from.

        Returns
        -------
        pd.DataFrame
            The data with zero columns removed.
        """
        return data.loc[:, (data != 0).any(axis=0)]


class DuibhirGrowthRatesPreprocessor(Preprocessor):

    _knockout_id = "knockout_id"
    _growth_rate_col = "growth_rate"

    def __init__(
        self,
        growth_rates_gateway: gateways.DuibhirGrowthRatesGateway,
        gene_names_mapper: mappers.Mapper,
    ):
        self.growth_rates_gateway = growth_rates_gateway
        self.gene_names_mapper = gene_names_mapper

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the growth rates data.

        Returns
        -------
        pd.DataFrame
            The preprocessed growth rates data.
        """

        data = self.growth_rates_gateway.get()

        data.rename(
            columns={"Row": self._knockout_id, "log2relT": self._growth_rate_col},
            inplace=True,
        )
        data = self.gene_names_mapper.map(
            data=data,
            target_col=self._knockout_id,
            new_col=self._knockout_id,
            fillna_col=self._knockout_id,
        )

        data.set_index(self._knockout_id, inplace=True)
        return data
