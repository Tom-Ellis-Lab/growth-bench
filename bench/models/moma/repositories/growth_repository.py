import abc

from bench.models.moma.gateways import growth_gateway
from bench.models.moma.entities import growth, omics


class GrowthRepositoryInterface(abc.ABC):
    @abc.abstractmethod
    def get(self) -> list[growth.GrowthRateMeasurement]:
        pass


class GrowthRepository(GrowthRepositoryInterface):
    def __init__(self, gateway: growth_gateway.GrowthDataGateway):
        self.gateway = gateway

    def get(self) -> list[growth.GrowthRateMeasurement]:
        growth_rate_data = self.gateway.get()
        result = []
        id_counter = 0
        for measurement in growth_rate_data:
            id_counter = id_counter + 1
            result.append(
                growth.GrowthRateMeasurement(
                    id=str(id_counter),
                    growth_rate=measurement.growth_rate,
                    condition=omics.GeneKnockout(
                        standard_name=measurement.ko_gene_standard_name,
                    ),
                    medium=measurement.medium,
                )
            )

        return result
