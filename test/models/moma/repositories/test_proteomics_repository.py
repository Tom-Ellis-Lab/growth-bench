from unittest import mock

import pytest

from bench.models.moma.entities import omics, proteomics
from bench.models.moma.gateways import proteomics_gateway
from bench.models.moma.repositories import proteomics_repository


class TestProteomicsRepository:

    @pytest.fixture
    def proteomics_data(self):
        result = [
            proteomics.ProteinProfileData(
                protein_id="A5Z2X5",
                protein_abundance_level=1.1,
                injection_nr=1,
                well_nr=2,
                batch_nr="hpr3",
                is_quality_control=False,
                is_his3delta_control=False,
                ko_gene_systematic_name="YAL001C",
                ko_gene_standard_name="TFC3",
                ko_gene_expression_level=0.5,
            ),
            proteomics.ProteinProfileData(
                protein_id="D6VTK4",
                protein_abundance_level=2.2,
                injection_nr=1,
                well_nr=2,
                batch_nr="hpr4",
                is_quality_control=True,
                is_his3delta_control=False,
                ko_gene_systematic_name="YAL002W",
                ko_gene_standard_name="TFC4",
                ko_gene_expression_level=0.6,
            ),
            proteomics.ProteinProfileData(
                protein_id="O13297",
                protein_abundance_level=3.3,
                injection_nr=1,
                well_nr=2,
                batch_nr="hpr5",
                is_quality_control=False,
                is_his3delta_control=True,
                ko_gene_systematic_name="YAL003W",
                ko_gene_standard_name="TFC5",
                ko_gene_expression_level=0.7,
            ),
        ]
        return result

    @pytest.fixture
    def gateway(self, proteomics_data):
        result = mock.Mock(spec=proteomics_gateway.ProteomicsGateway)
        result.get = mock.Mock(return_value=proteomics_data)
        return result

    @pytest.fixture
    def expected(self, proteomics_data):
        result = [
            proteomics.ProteinAbundanceProfile(
                id=str(count + 1),
                protein=proteomics.Protein(
                    id=i.protein_id,
                ),
                abundance_level=i.protein_abundance_level,
                condition=omics.GeneKnockout(
                    standard_name=i.ko_gene_standard_name,
                    systematic_name=i.ko_gene_systematic_name,
                    expression_level=i.ko_gene_expression_level,
                ),
                metadata=proteomics.ProteomicsMetadata(
                    injection_nr=i.injection_nr,
                    well_nr=i.well_nr,
                    batch_nr=i.batch_nr,
                    is_quality_control=i.is_quality_control,
                    is_his3delta_control=i.is_his3delta_control,
                ),
            )
            for count, i in enumerate(proteomics_data)
        ]
        return result

    def test_get(self, gateway, expected):
        observed = proteomics_repository.ProteomicsRepository(gateway=gateway).get()

        assert len(observed) == len(expected)
        for obs, exp in zip(observed, expected):
            assert obs.id == exp.id
            assert obs.molecular_entity.id == exp.molecular_entity.id
            assert obs.value == exp.value
            assert (
                obs.metadata_details.injection_nr == exp.metadata_details.injection_nr
            )
            assert obs.metadata_details.well_nr == exp.metadata_details.well_nr
            assert obs.metadata_details.batch_nr == exp.metadata_details.batch_nr
            assert obs.condition.name == exp.condition.name
