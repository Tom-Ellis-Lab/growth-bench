import pathlib
from unittest import mock

import pandas as pd
import pytest

from bench.models.moma.gateways import transcriptomics_gateway
from bench.models.moma.entities import transcriptomics


class TestTranscriptomicsGateway:

    @pytest.fixture
    def file_path(self):
        result = pathlib.Path("proteomics.RDS")
        return result

    @pytest.fixture
    def culley_data_file_path(self):
        result = pathlib.Path("culley_data.RDS")
        return result

    @pytest.fixture
    def transcriptomics(self):
        result = pd.DataFrame(
            {
                "YWL016": [1.1, 2.2, 3.3],
                "log2relT": [1.2, 3.4, 5.6],
            }
        )
        return result

    @pytest.fixture
    def culley_data(self):
        result = pd.DataFrame(
            {
                "Row": ["SPO7", "SWC3", "DEP1"],
            }
        )
        return result

    @pytest.fixture
    def expected(self):
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

    def test_get(
        self, file_path, culley_data_file_path, transcriptomics, culley_data, expected
    ):
        with mock.patch("pyreadr.read_r") as mock_read_r:
            mock_read_r.side_effect = [
                {None: transcriptomics},
                {None: culley_data},
            ]
            gateway = transcriptomics_gateway.TranscriptomicsGateway(
                file_path=file_path, culley_data_file_path=culley_data_file_path
            )
            observed = gateway.get()

            assert observed == expected
