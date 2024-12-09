import abc
import csv
import pathlib

import pyreadr

from bench.models.moma.entities import growth


class GrowthDataGateway(abc.ABC):

    @abc.abstractmethod
    def get(self) -> list[growth.GrowthRateData]:
        pass


class DuibhirGrowthDataGateway(GrowthDataGateway):

    _headers = {"knockout_id": "Row", "growth_rate": "log2relT"}

    def __init__(self, file_path: pathlib.Path):
        self._file_path = file_path

    @property
    def file_path(self):
        return self._file_path

    def get(self) -> list[growth.GrowthRateData]:
        data = pyreadr.read_r(self.file_path)[None]

        result = []
        for _, row in data.iterrows():
            ko_gene_standard_name = row[self._headers["knockout_id"]]
            growth_rate = row[self._headers["growth_rate"]]
            result.append(
                growth.GrowthRateData(
                    growth_rate=growth_rate,
                    ko_gene_standard_name=ko_gene_standard_name,
                    medium=growth.GrowthMedium.SD,
                )
            )

        return result


class Yeast5kGrowthDataGateway(GrowthDataGateway):

    _headers = {
        "ko_gene_standard_name": "standard_name",
        "ko_gene_systematic_name": "systematic_name",
        growth.GrowthMedium.YPD: "YPD",
        growth.GrowthMedium.SM: "SM",
        growth.GrowthMedium.SC: "SC",
    }

    def __init__(self, file_path: pathlib.Path, growth_medium: growth.GrowthMedium):
        self._file_path = file_path
        self._growth_medium = growth_medium

    @property
    def file_path(self) -> pathlib.Path:
        return self._file_path

    @property
    def growth_medium(self) -> growth.GrowthMedium:
        return self._growth_medium

    def get(self) -> list[growth.GrowthRateData]:

        result = []
        with open(self.file_path, mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                standard_name = row[self._headers["ko_gene_standard_name"]]
                systematic_name = row[self._headers["ko_gene_systematic_name"]]
                growth_rate = float(row[self._headers[self.growth_medium]])

                result.append(
                    growth.GrowthRateData(
                        growth_rate=growth_rate,
                        ko_gene_standard_name=standard_name,
                        ko_gene_systematic_name=systematic_name,
                        medium=self.growth_medium,
                    )
                )

        return result
