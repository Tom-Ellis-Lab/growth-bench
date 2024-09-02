import dataclasses
from typing import Optional

from bench.models.moma.entities import omics


@dataclasses.dataclass
class Protein(omics.MolecularEntity):
    pass


@dataclasses.dataclass
class ProteinProfileData:
    protein_id: str
    protein_abundance_level: float
    injection_nr: int
    well_nr: int
    batch_nr: str
    is_quality_control: bool
    is_his3delta_control: bool
    ko_gene_systematic_name: str
    ko_gene_standard_name: str
    ko_gene_expression_level: Optional[float] = None


@dataclasses.dataclass
class ProteomicsMetadata:
    injection_nr: int
    well_nr: int
    batch_nr: str
    is_quality_control: bool
    is_his3delta_control: bool


class ProteinAbundanceProfile(omics.QuantitativeMeasurement):
    def __init__(
        self,
        id: str,
        protein: Protein,
        condition: omics.ExperimentalCondition,
        abundance_level: float,
        metadata: ProteomicsMetadata,
    ):
        self._id = id
        self._protein = protein
        self._condition = condition
        self._abundance_level = abundance_level
        self._metadata = metadata

    @property
    def id(self) -> str:
        return self._id

    @property
    def molecular_entity(self) -> Protein:
        return self._protein

    @property
    def value(self) -> float:
        return self._abundance_level

    @property
    def condition(self) -> omics.ExperimentalCondition:
        return self._condition

    @property
    def metadata_details(self) -> ProteomicsMetadata:
        return self._metadata

    @property
    def quality_control(self) -> bool:
        return self._metadata.is_quality_control

    @property
    def his3delta_control(self) -> bool:
        return self._metadata.is_his3delta_control


class Proteomics(omics.Omics):
    def __init__(self, data: list[ProteinAbundanceProfile]):
        self._data = data

    def return_as_list(self) -> list[ProteinAbundanceProfile]:
        return self._data

    def get_by_id(self, id: str) -> ProteinAbundanceProfile:
        for profile in self._data:
            if profile.id == id:
                return profile
        raise ValueError(f"No profile with id '{id}' found.")

    def group_by_id(
        self,
    ) -> dict[str, list[ProteinAbundanceProfile]]:
        result = {}

        for profile in self._data:
            if profile.molecular_entity.id not in result:
                result[profile.molecular_entity.id] = []
            molecular_id = profile.molecular_entity.id
            result[molecular_id].append(profile)

        return result

    def group_by_condition(self) -> dict[str, list[ProteinAbundanceProfile]]:
        result = {}
        for profile in self._data:
            condition = profile.condition
            if condition.name not in result:
                result[condition.name] = []
            result[condition.name].append(profile)
        return result

    def set(self, profiles: list[ProteinAbundanceProfile]) -> "Proteomics":
        self._data = profiles
        return self
