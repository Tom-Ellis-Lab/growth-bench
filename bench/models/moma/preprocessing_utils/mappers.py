import abc
import enum

import pandas as pd

from bench.models.moma.gateways import gateways


class MapperType(enum.Enum):
    GENE_NAMES = "gene_names"


class MapperFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_mapper(self, mapper_type: "MapperType") -> "Mapper":
        """
        Create a mapper based on the mapper type.

        Parameters
        ----------
        mapper_type : MapperType
            The type of mapper to create.

        Returns
        -------
        Mapper
            The mapper.
        """


class MapperFactory(MapperFactoryInterface):

    def create_mapper(self, mapper_type: "MapperType") -> "Mapper":
        """
        Create a mapper based on the mapper type.

        Parameters
        ----------
        mapper_type : MapperType
            The type of mapper to create.

        Returns
        -------
        Mapper
            The mapper.
        """
        if mapper_type == MapperType.GENE_NAMES:
            return GeneNamesMapper()
        else:
            raise ValueError(f"Unknown mapper type: {mapper_type}")


class Mapper(abc.ABC):
    @abc.abstractmethod
    def map(
        self, data: pd.DataFrame, target_col: str, new_col: str, fillna_col: str
    ) -> pd.DataFrame:
        """
        Map the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data to map.
        target_col: str
            The target column to map.
        new_col: str
            The new column to create.
        fillna_col: str
            The column to fillna.

        Returns
        -------
        pd.DataFrame
            The mapped data.
        """
        pass

    @abc.abstractmethod
    def create_mapping(
        self, gene_names_gateway: gateways.GeneNamesGateway
    ) -> dict[str, str]:
        """
        Create a mapping

        Returns
        -------
        Any
            The data with the string converted to another string.
        """
        pass


class GeneNamesMapper(Mapper):
    def __init__(self):
        self._mapping_dict = None

    def map(
        self, data: pd.DataFrame, target_col: str, new_col: str, fillna_col: str
    ) -> pd.DataFrame:
        """
        Map gene names.

        Parameters
        ----------
        data : pd.DataFrame
            The data to map.
        target_col: str
            The target column to map.
        new_col: str
            The new column to create.
        fillna_col: str
            The column to fillna.

        Returns
        -------
        dict[str, str]
            The gene names mapping.
        """
        if self._mapping_dict is None:
            raise ValueError("Mapping dict not set")

        data[new_col] = (
            data[target_col].map(self._mapping_dict).fillna(data[fillna_col])
        )
        return data

    def create_mapping(
        self, gene_names_gateway: gateways.GeneNamesGateway
    ) -> dict[str, str]:
        """
        Create a mapping between standard and systematic names

        Returns
        -------
        pd.DataFrame
            The data with the standard names converted to systematic names.
        """
        gene_names = gene_names_gateway.get()
        systematic_name = gene_names_gateway.get_systematic_name()
        standard_name = gene_names_gateway.get_standard_name()
        result = pd.Series(
            gene_names[systematic_name].values,
            index=gene_names[standard_name],
        ).to_dict()
        # Also add systematic names as keys mapping to themselves
        result.update(
            pd.Series(
                gene_names[systematic_name].values,
                index=gene_names[systematic_name],
            ).to_dict()
        )
        self._mapping_dict = result
        return result
