import abc
import dataclasses
import enum
from typing import Optional

from bench.models.moma.entities import omics


class GrowthMedium(enum.Enum):
    SC = "SC"
    SM = "SM"
    YPD = "YPD"
    SD = "SD"


@dataclasses.dataclass
class GrowthRateData:
    growth_rate: float
    medium: GrowthMedium
    ko_gene_standard_name: Optional[str] = None
    ko_gene_systematic_name: Optional[str] = None

    def __post_init__(self):
        if not self.ko_gene_standard_name and not self.ko_gene_systematic_name:
            raise ValueError(
                "Either ko_gene_standard_name or ko_gene_systematic_name must be provided"
            )


class GrowthRateMeasurement:
    def __init__(
        self,
        id: str,
        growth_rate: float,
        condition: omics.ExperimentalCondition,
        medium,
    ):
        self._id = id
        self._growth_rate = growth_rate
        self._condition = condition
        self._medium = medium

    @property
    def id(self) -> str:
        return self._id

    @property
    def growth_rate(self) -> float:
        return self._growth_rate

    @property
    def condition(self) -> omics.ExperimentalCondition:
        return self._condition

    @property
    def medium(self) -> GrowthMedium:
        return self._medium


class GrowthRateDatasetInterface(abc.ABC):

    @abc.abstractmethod
    def get_by_id(self, id: str) -> GrowthRateMeasurement:
        pass

    @abc.abstractmethod
    def group_by_id(self) -> dict[str, list[GrowthRateMeasurement]]:
        pass

    @abc.abstractmethod
    def group_by_condition(self) -> dict[str, list[GrowthRateMeasurement]]:
        pass

    @abc.abstractmethod
    def group_by_medium(self) -> dict[str, list[GrowthRateMeasurement]]:
        pass


class GrowthRateDataset:
    def __init__(self, data: list[GrowthRateMeasurement]):
        self._data = data

    def get_by_id(self, id: str) -> GrowthRateMeasurement:
        for profile in self._data:
            if profile.id == id:
                return profile
        raise ValueError(f"Profile with id {id} not found")

    def group_by_id(self) -> dict[str, list[GrowthRateMeasurement]]:
        result = {}
        for profile in self._data:
            if profile.id not in result:
                result[profile.id] = []
            result[profile.id].append(profile)
        return result

    def group_by_condition(self) -> dict[str, list[GrowthRateMeasurement]]:
        result = {}
        for profile in self._data:
            condition = profile.condition
            if condition.name not in result:
                result[condition.name] = []
            result[condition.name].append(profile)
        return result

    def group_by_medium(self) -> dict[str, list[GrowthRateMeasurement]]:
        result = {}
        for profile in self._data:
            medium = profile.medium
            if medium.name not in result:
                result[medium.name] = []
            result[medium.name].append(profile)
        return result
