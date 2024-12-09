import dataclasses

from bench.models.moma.entities import omics


@dataclasses.dataclass
class Transcript(omics.MolecularEntity):
    pass


@dataclasses.dataclass
class TranscriptProfileData:
    transcript_id: str
    transcript_expression_level: float
    ko_gene_standard_name: str


class TranscriptExpressionProfile(omics.QuantitativeMeasurement):
    def __init__(
        self,
        id: str,
        transcript: Transcript,
        condition: omics.ExperimentalCondition,
        expression_level: float,
    ):
        self._id = id
        self._transcript = transcript
        self._condition = condition
        self._expression_level = expression_level

    @property
    def id(self) -> str:
        return self._id

    @property
    def molecular_entity(self) -> Transcript:
        return self._transcript

    @property
    def value(self) -> float:
        return self._expression_level

    @property
    def condition(self) -> omics.ExperimentalCondition:
        # TOOD: rename _condition to _experimental_condition
        return self._condition

    @property
    def quality_control(self) -> bool:
        raise NotImplementedError

    @property
    def his3delta_control(self) -> bool:
        raise NotImplementedError


class Transcriptomics(omics.Omics):

    def __init__(self, data: list["TranscriptExpressionProfile"]):
        self._data = data

    def return_as_list(self) -> list["TranscriptExpressionProfile"]:
        return self._data

    def get_by_id(self, id: str) -> "TranscriptExpressionProfile":
        for profile in self._data:
            if profile.id == id:
                return profile
        raise ValueError(f"Transcriptomics profile with id {id} not found")

    def group_by_id(
        self,
    ) -> dict[str, list["TranscriptExpressionProfile"]]:
        result = {}
        for profile in self._data:
            if profile.molecular_entity.id not in result:
                result[profile.molecular_entity.id] = []
            result[profile.molecular_entity.id].append(profile)
        return result

    def group_by_condition(self) -> dict[str, list["TranscriptExpressionProfile"]]:
        result = {}
        for profile in self._data:
            condition = profile.condition
            if condition.name not in result:
                result[condition.name] = []
            result[condition.name].append(profile)
        return result

    def set(self, profiles: list["TranscriptExpressionProfile"]) -> "Transcriptomics":
        self._data = profiles
        return self
