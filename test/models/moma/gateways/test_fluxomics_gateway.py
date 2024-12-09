import pathlib
from unittest import mock

import pandas as pd
import pytest

from bench.models.moma.gateways import fluxomics_gateway
from bench.models.moma.entities import fluxomics


class TestFluxomicsGateway:

    @pytest.fixture
    def transcriptomics_file_path(self):
        result = pathlib.Path("transcriptoimcs.RDS")
        return result

    @pytest.fixture
    def culley_data_file_path(self):
        result = pathlib.Path("culley_data.RDS")
        return result

    @pytest.fixture
    def transcriptomics(self):
        result = pd.DataFrame(
            {
                "log2relT": [1.2, 3.4, 5.6],
                "YDL004W": [1.1, 2.2, 3.3],
                "YFG04W": [4.4, 5.5, 6.6],
            }
        )
        return result

    @pytest.fixture
    def culley_data(self):
        result = pd.DataFrame(
            {
                "Row": ["SPO7", "SWC3", "DEP1"],
                "log2relT": [1.2, 3.4, 5.6],
                "YDL004W": [1.1, 2.2, 3.3],
                "YFG04W": [4.4, 5.5, 6.6],
                "r_0003": [1.1, 2.2, 3.3],
            }
        )
        return result

    @pytest.fixture
    def expected(self):
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

    def test_get(
        self,
        transcriptomics_file_path,
        culley_data_file_path,
        transcriptomics,
        culley_data,
        expected,
    ):
        with mock.patch("pyreadr.read_r") as mock_read_r:
            mock_read_r.side_effect = [
                {None: transcriptomics},
                {None: culley_data},
            ]
            gateway = fluxomics_gateway.FluxomicsGateway(
                transcriptomics_file_path=transcriptomics_file_path,
                culley_data_file_path=culley_data_file_path,
            )
            observed = gateway.get()
            assert observed == expected
