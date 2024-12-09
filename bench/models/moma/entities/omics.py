import abc
import dataclasses
from typing import Optional


@dataclasses.dataclass
class MolecularEntity(abc.ABC):
    id: str


class ExperimentalCondition(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class GeneKnockout(ExperimentalCondition):

    def __init__(
        self,
        standard_name: Optional[str] = None,
        systematic_name: Optional[str] = None,
        expression_level: Optional[float] = None,
    ):
        self._verify_names(standard_name=standard_name, systematic_name=systematic_name)
        self._standard_name = standard_name
        self._systematic_name = systematic_name
        self._expression_level = expression_level

    def _verify_names(self, standard_name, systematic_name):
        if not standard_name and not systematic_name:
            raise ValueError("Either standard_name or systematic_name must be provided")

    @property
    def name(self) -> str:
        if self._standard_name:
            return self._standard_name
        elif self._systematic_name:
            return self._systematic_name
        else:
            raise ValueError("Either standard_name or systematic_name must be provided")


class QuantitativeMeasurement(abc.ABC):

    @property
    @abc.abstractmethod
    def id(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def molecular_entity(self) -> "MolecularEntity":
        pass

    @property
    @abc.abstractmethod
    def value(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def condition(self) -> "ExperimentalCondition":
        pass

    @property
    @abc.abstractmethod
    def quality_control(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def his3delta_control(self) -> bool:
        pass

    @property
    def type(self) -> str:
        return self.__class__.__name__


class Omics(abc.ABC):

    @abc.abstractmethod
    def __init__(self, data: list["QuantitativeMeasurement"]):
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @abc.abstractmethod
    def return_as_list(self) -> list["QuantitativeMeasurement"]:
        pass

    @abc.abstractmethod
    def get_by_id(self, id: str) -> "QuantitativeMeasurement":
        pass

    @abc.abstractmethod
    def group_by_id(
        self,
    ) -> dict[str, list["QuantitativeMeasurement"]]:
        pass

    @abc.abstractmethod
    def group_by_condition(
        self,
    ) -> dict[str, list["QuantitativeMeasurement"]]:
        pass

    @abc.abstractmethod
    def set(self, profiles: list["QuantitativeMeasurement"]) -> "Omics":
        pass
