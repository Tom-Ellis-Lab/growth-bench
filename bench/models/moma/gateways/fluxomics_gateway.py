import abc
import pathlib

import pyreadr

from bench.models.moma.entities import fluxomics


class FluxomicsGatewayInterface(abc.ABC):
    @abc.abstractmethod
    def get(self) -> fluxomics.MetabolicReactionData:
        pass


class FluxomicsGateway(FluxomicsGatewayInterface):

    _knockout_id_header = "Row"

    def __init__(
        self,
        transcriptomics_file_path: pathlib.Path,
        culley_data_file_path: pathlib.Path,
    ):
        self._transcriptomics_file_path = transcriptomics_file_path
        self._culley_data_file_path = culley_data_file_path

    @property
    def transcriptomics_file_path(self) -> pathlib.Path:
        return self._transcriptomics_file_path

    @property
    def culley_data_file_path(self) -> pathlib.Path:
        return self._culley_data_file_path

    def get(self) -> list[fluxomics.MetabolicReactionData]:

        transcriptomics = pyreadr.read_r(self.transcriptomics_file_path)[None]
        culley_data = pyreadr.read_r(self.culley_data_file_path)[None]
        data = culley_data.drop(columns=transcriptomics.columns.values)

        result = []
        for _, row in data.iterrows():
            ko_gene_standard_name = row[self._knockout_id_header]
            for reaction_id, flux_rate in row.items():
                if reaction_id == self._knockout_id_header:
                    continue
                flux_profile = fluxomics.MetabolicReactionData(
                    reaction_id=reaction_id,
                    flux_rate=float(flux_rate),
                    ko_gene_standard_name=ko_gene_standard_name,
                )
                result.append(flux_profile)

        return result
