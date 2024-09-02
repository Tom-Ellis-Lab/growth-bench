import abc
from typing import Callable

from bench.models.moma.entities import omics


# TODO: Explain how this module should be used


class OmicsFilteringStrategy(abc.ABC):

    @abc.abstractmethod
    def get_strategy(
        self,
    ) -> Callable[[omics.Omics], omics.Omics]:
        pass


class RemoveQualityControl(OmicsFilteringStrategy):

    def get_strategy(
        self,
    ) -> Callable[[omics.Omics], omics.Omics]:
        def strategy(
            data: omics.Omics,
        ) -> omics.Omics:
            filtered = [
                measurement for measurement in data if not measurement.quality_control
            ]
            result = data.set(profiles=filtered)
            return result

        return strategy


class RemoveHis3deltaControl(OmicsFilteringStrategy):

    def get_strategy(
        self,
    ) -> Callable[[omics.Omics], omics.Omics]:

        def strategy(
            data: omics.Omics,
        ) -> omics.Omics:
            filtered = [
                measurement for measurement in data if not measurement.his3delta_control
            ]
            result = data.set(profiles=filtered)
            return result

        return strategy


class RemoveZeroSamples(OmicsFilteringStrategy):

    def get_strategy(
        self,
    ) -> Callable[[omics.Omics], omics.Omics]:

        def strategy(
            data: omics.Omics,
        ) -> omics.Omics:
            grouped_omics = data.group_by_id()
            filtered = []
            for protein_name, measurements in grouped_omics.items():
                if not all([measurement.value == 0 for measurement in measurements]):
                    filtered.extend(measurements)
            result = data.set(profiles=filtered)
            return result

        return strategy


class OmicsFilter(abc.ABC):

    @abc.abstractmethod
    def filter(self, data: omics.Omics) -> omics.Omics:
        pass


class Filter(OmicsFilter):

    def __init__(
        self,
        strategies: list[OmicsFilteringStrategy],
    ):
        self._strategies = strategies

    def filter(
        self,
        data: omics.Omics,
    ) -> omics.Omics:
        result = data
        for strategy in self._strategies:
            filter = strategy.get_strategy()
            result = filter(result)
        return result


class OmicsFilterFactory(abc.ABC):

    @abc.abstractmethod
    def create_filter(self) -> OmicsFilter:
        pass


class FilterFactory(OmicsFilterFactory):

    def __init__(
        self,
        strategies: list[OmicsFilteringStrategy],
    ):
        self._strategies = strategies

    def create_filter(self) -> OmicsFilter:
        return Filter(strategies=self._strategies)
