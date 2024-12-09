from unittest import mock

import pytest

from bench.models.moma.entities import omics, transcriptomics
from bench.models.moma.gateways import transcriptomics_gateway
from bench.models.moma.repositories import transcriptomics_repository


class TestTranscriptomicsRepository:

    @pytest.fixture
    def transcriptomics_data(self):
        result = [
            transcriptomics.TranscriptProfileData(
                transcript_id="YWL016",
                transcript_expression_level=1.1,
                ko_gene_standard_name="SPO7",
            ),
            transcriptomics.TranscriptProfileData(
                transcript_id="YWL016",
                transcript_expression_level=2.2,
                ko_gene_standard_name="SWC3",
            ),
            transcriptomics.TranscriptProfileData(
                transcript_id="YWL016",
                transcript_expression_level=3.3,
                ko_gene_standard_name="DEP1",
            ),
        ]
        return result

    @pytest.fixture
    def expected(self, transcriptomics_data):
        result = [
            transcriptomics.TranscriptExpressionProfile(
                id=str(count + 1),
                transcript=transcriptomics.Transcript(id=i.transcript_id),
                condition=omics.GeneKnockout(standard_name=i.ko_gene_standard_name),
                expression_level=i.transcript_expression_level,
            )
            for count, i in enumerate(transcriptomics_data)
        ]
        return result

    @pytest.fixture
    def gateway(self, transcriptomics_data):
        result = mock.Mock(spec=transcriptomics_gateway.TranscriptomicsGateway)
        result.get = mock.Mock(return_value=transcriptomics_data)
        return result

    def test_get(self, gateway, expected):
        observed = transcriptomics_repository.TranscriptomicsRepository(
            gateway=gateway
        ).get()
        assert len(observed) == len(expected)
        for obs, exp in zip(observed, expected):
            assert obs.id == exp.id
            assert obs.molecular_entity.id == exp.molecular_entity.id
            assert obs.condition.name == exp.condition.name
            assert obs.value == exp.value
