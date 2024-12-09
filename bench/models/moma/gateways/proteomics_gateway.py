import abc
import csv

import pathlib

from bench.models.moma.entities import proteomics


class ProteomicsGatewayInterface(abc.ABC):
    @abc.abstractmethod
    def get(self) -> proteomics.ProteinProfileData:
        pass


class ProteomicsGateway(ProteomicsGatewayInterface):
    _old_header = "Protein.Group"
    _protein_id_header = "Protein.Group"
    _qc_marker = "qc"
    _his3_marker = "HIS3"

    def __init__(self, file_path: pathlib.Path):
        self._file_path = file_path

    @property
    def file_path(self) -> pathlib.Path:
        return self._file_path

    def get(self) -> list[proteomics.ProteinProfileData]:
        with open(self.file_path, "r") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("The CSV file does not contain headers.")
            headers = list(fieldnames)
            if self._old_header not in headers:
                raise ValueError(
                    f"Expected {self._old_header} in headers but got {headers}"
                )

            result = []
            for row in reader:
                protein_id = row.get(self._old_header, "")

                for header, value in row.items():

                    if header == self._old_header:
                        continue

                    metadata = header.split("_")

                    data = proteomics.ProteinProfileData(
                        protein_id=protein_id,
                        protein_abundance_level=float(value),
                        injection_nr=int(metadata[0]),
                        well_nr=int(metadata[1]),
                        batch_nr=metadata[2],
                        is_quality_control=metadata[3] == self._qc_marker,
                        is_his3delta_control=metadata[3] == self._his3_marker,
                        ko_gene_systematic_name=metadata[4],
                        ko_gene_standard_name=metadata[5],
                        ko_gene_expression_level=(
                            None if metadata[6] == "NA" else float(metadata[6])
                        ),
                    )
                    result.append(data)

        return result
