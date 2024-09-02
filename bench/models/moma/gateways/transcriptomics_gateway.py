import abc
import pathlib

import pyreadr

from bench.models.moma.entities import transcriptomics


class TranscriptomicsGatewayInterface(abc.ABC):
    @abc.abstractmethod
    def get(self) -> transcriptomics.TranscriptProfileData:
        pass


class TranscriptomicsGateway(TranscriptomicsGatewayInterface):

    _growth_rate_header = "log2relT"
    _knockout_id_header = "knockout_id"
    _headers = {_knockout_id_header: "Row"}

    def __init__(self, file_path: pathlib.Path, culley_data_file_path: pathlib.Path):
        self._file_path = file_path
        self._culley_data_file_path = culley_data_file_path

    @property
    def file_path(self) -> pathlib.Path:
        return self._file_path

    @property
    def culley_data_file_path(self) -> pathlib.Path:
        return self._culley_data_file_path

    def get(self) -> list[transcriptomics.TranscriptProfileData]:
        data = pyreadr.read_r(path=self.file_path)[None]
        culley_data = pyreadr.read_r(path=self.culley_data_file_path)[None]
        data.drop(columns=self._growth_rate_header, inplace=True)
        data[self._knockout_id_header] = culley_data[
            self._headers[self._knockout_id_header]
        ]
        result = []
        # Iterate over rows in the data
        for _, row in data.iterrows():
            ko_gene_standard_name = row[self._knockout_id_header]
            for transcript_id, expression_level in row.items():
                if transcript_id == self._knockout_id_header:
                    continue
                transcript_profile = transcriptomics.TranscriptProfileData(
                    transcript_id=transcript_id,
                    transcript_expression_level=float(expression_level),
                    ko_gene_standard_name=ko_gene_standard_name,
                )
                result.append(transcript_profile)

        return result
