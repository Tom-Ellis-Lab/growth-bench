import pathlib
from unittest import mock

import pytest

from bench.models.moma.entities import growth, omics
from bench.models.moma.gateways import growth_gateway
from bench.models.moma.repositories import growth_repository


class TestYeast5kGrowthRepository:

    @pytest.fixture
    def growth_data(self):
        result = [
            growth.GrowthRateData(
                growth_rate=1.2,
                ko_gene_systematic_name="YAL001C",
                ko_gene_standard_name="FLO9",
                medium=growth.GrowthMedium.YPD,
            ),
            growth.GrowthRateData(
                growth_rate=3.4,
                ko_gene_systematic_name="YBL002W",
                ko_gene_standard_name="FLO10",
                medium=growth.GrowthMedium.SM,
            ),
            growth.GrowthRateData(
                growth_rate=5.6,
                ko_gene_systematic_name="YCL003W",
                ko_gene_standard_name="FLO11",
                medium=growth.GrowthMedium.SD,
            ),
        ]
        return result

    @pytest.fixture
    def gateway(self, growth_data):
        result = mock.Mock(spec=growth_gateway.Yeast5kGrowthDataGateway)
        result.get = mock.Mock(return_value=growth_data)
        return result

    @pytest.fixture
    def expected(self):
        result = [
            growth.GrowthRateMeasurement(
                id="1",
                growth_rate=1.2,
                condition=omics.GeneKnockout(
                    standard_name="FLO9",
                    systematic_name="YAL001C",
                ),
                medium=growth.GrowthMedium.YPD,
            ),
            growth.GrowthRateMeasurement(
                id="2",
                growth_rate=3.4,
                condition=omics.GeneKnockout(
                    standard_name="FLO10",
                    systematic_name="YBL002W",
                ),
                medium=growth.GrowthMedium.SM,
            ),
            growth.GrowthRateMeasurement(
                id="3",
                growth_rate=5.6,
                condition=omics.GeneKnockout(
                    standard_name="FLO11",
                    systematic_name="YCL003W",
                ),
                medium=growth.GrowthMedium.SD,
            ),
        ]
        return result

    def test_get(self, gateway, expected):
        observed = growth_repository.GrowthRepository(gateway=gateway).get()
        assert len(observed) == len(expected)
        for i, j in zip(observed, expected):
            assert i.id == j.id
            assert i.growth_rate == j.growth_rate
            assert i.condition.name == j.condition.name
            assert i.medium == j.medium
