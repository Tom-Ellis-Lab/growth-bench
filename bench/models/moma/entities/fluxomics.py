import dataclasses

from bench.models.moma.entities import omics


@dataclasses.dataclass
class MetabolicReaction(omics.MolecularEntity):
    pass


@dataclasses.dataclass
class MetabolicReactionData:
    reaction_id: str
    flux_rate: float
    ko_gene_standard_name: str


class MetabolicFLuxProfile(omics.QuantitativeMeasurement):

    def __init__(
        self,
        id: str,
        reaction: "MetabolicReaction",
        condition: omics.ExperimentalCondition,
        flux_rate: float,
    ):
        self._id = id
        self._reaction = reaction
        self._condition = condition
        self._flux_rate = flux_rate

    @property
    def id(self) -> str:
        return self._id

    @property
    def molecular_entity(self) -> "MetabolicReaction":
        return self._reaction

    @property
    def value(self) -> float:
        return self._flux_rate

    @property
    def condition(self) -> omics.ExperimentalCondition:
        return self._condition

    @property
    def quality_control(self) -> bool:
        raise NotImplementedError

    @property
    def his3delta_control(self) -> bool:
        raise NotImplementedError


class Fluxomics(omics.Omics):
    def __init__(self, data: list["MetabolicFLuxProfile"]):
        self._data = data

    def return_as_list(self) -> list["MetabolicFLuxProfile"]:
        return self._data

    def get_by_id(self, id: str) -> "MetabolicFLuxProfile":
        for profile in self._data:
            if profile.id == id:
                return profile
        raise ValueError(f"Profile with id {id} not found")

    def group_by_id(
        self,
    ) -> dict[str, list["MetabolicFLuxProfile"]]:
        result = {}
        for profile in self._data:
            if profile.molecular_entity.id not in result:
                result[profile.molecular_entity.id] = []
            result[profile.molecular_entity.id].append(profile)
        return result

    def group_by_condition(self) -> dict[str, list["MetabolicFLuxProfile"]]:
        result = {}
        for profile in self._data:
            condition = profile.condition
            if condition.name not in result:
                result[condition.name] = []
            result[condition.name].append(profile)
        return result

    def set(self, profiles: list["MetabolicFLuxProfile"]) -> "Fluxomics":
        self._data = profiles
        return self
