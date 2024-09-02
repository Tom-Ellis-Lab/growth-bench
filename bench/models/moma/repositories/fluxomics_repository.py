import abc

from bench.models.moma.gateways import fluxomics_gateway
from bench.models.moma.entities import fluxomics, omics


class FluxomicsRepositoryInterface(abc.ABC):
    @abc.abstractmethod
    def get(self) -> list[fluxomics.MetabolicFLuxProfile]:
        pass


class FluxomicsRepository(FluxomicsRepositoryInterface):
    def __init__(self, gateway: fluxomics_gateway.FluxomicsGateway):
        self.gateway = gateway

    def get(self) -> list[fluxomics.MetabolicFLuxProfile]:
        data = self.gateway.get()

        result = []
        id_counter = 0
        for flux_profile in data:
            id_counter = id_counter + 1
            profile = fluxomics.MetabolicFLuxProfile(
                id=str(id_counter),
                reaction=fluxomics.MetabolicReaction(id=flux_profile.reaction_id),
                condition=omics.GeneKnockout(
                    standard_name=flux_profile.ko_gene_standard_name
                ),
                flux_rate=flux_profile.flux_rate,
            )
            result.append(profile)

        return result
