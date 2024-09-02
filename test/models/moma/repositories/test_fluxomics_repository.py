from unittest import mock

import pytest

from bench.models.moma.entities import fluxomics, omics
from bench.models.moma.gateways import fluxomics_gateway
from bench.models.moma.repositories import fluxomics_repository


class TestFluxomicsRepository:

    @pytest.fixture
    def fluxomics_data(self):
        result = [
            fluxomics.MetabolicReactionData(
                reaction_id="r_0003",
                flux_rate=1.1,
                ko_gene_standard_name="SPO7",
            ),
            fluxomics.MetabolicReactionData(
                reaction_id="r_0003",
                flux_rate=2.2,
                ko_gene_standard_name="SWC3",
            ),
            fluxomics.MetabolicReactionData(
                reaction_id="r_0003",
                flux_rate=3.3,
                ko_gene_standard_name="DEP1",
            ),
        ]
        return result

    @pytest.fixture
    def gateway(self, fluxomics_data):
        result = mock.Mock(spec=fluxomics_gateway.FluxomicsGateway)
        result.get = mock.Mock(return_value=fluxomics_data)
        return result

    @pytest.fixture
    def expected(self, fluxomics_data):
        result = [
            fluxomics.MetabolicFLuxProfile(
                id=str(count + 1),
                reaction=fluxomics.MetabolicReaction(id=i.reaction_id),
                condition=omics.GeneKnockout(standard_name=i.ko_gene_standard_name),
                flux_rate=i.flux_rate,
            )
            for count, i in enumerate(fluxomics_data)
        ]
        return result

    def test_get(self, gateway, expected):
        observed = fluxomics_repository.FluxomicsRepository(gateway=gateway).get()
        assert len(observed) == len(expected)
        for obs, exp in zip(observed, expected):
            assert obs.id == exp.id
            assert obs.molecular_entity.id == exp.molecular_entity.id
            assert obs.condition.name == exp.condition.name
            assert obs.value == exp.value
