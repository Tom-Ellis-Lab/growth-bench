import pathlib
from unittest import mock

import pandas as pd
import pytest

from bench.models.moma.gateways import growth_gateway
from bench.models.moma.entities import growth


class TestDuibhirGrowthDataGateway:

    @pytest.fixture
    def file_path(self):
        result = pathlib.Path("growth_data.RDS")
        return result

    @pytest.fixture
    def growth_data(self):
        result = pd.DataFrame(
            {
                "Row": ["SPO7", "SWC3", "NTG1"],
                "log2relT": [1.2, 3.4, 5.6],
                "gene_1": [1.1, 2.2, 3.3],
                "gene_2": [4.4, 5.5, 6.6],
            }
        )
        return result

    @pytest.fixture
    def expected(self):
        result = [
            growth.GrowthRateData(
                growth_rate=1.2,
                ko_gene_standard_name="SPO7",
                medium=growth.GrowthMedium.SD,
            ),
            growth.GrowthRateData(
                growth_rate=3.4,
                ko_gene_standard_name="SWC3",
                medium=growth.GrowthMedium.SD,
            ),
            growth.GrowthRateData(
                growth_rate=5.6,
                ko_gene_standard_name="NTG1",
                medium=growth.GrowthMedium.SD,
            ),
        ]
        return result

    def test_get(self, file_path, growth_data, expected):
        with mock.patch("pyreadr.read_r") as mock_read_r:
            mock_read_r.side_effect = [
                {None: growth_data},
            ]
            gateway = growth_gateway.DuibhirGrowthDataGateway(file_path=file_path)
            observed = gateway.get()
        assert observed == expected


class TestYeast5kGrowthDataGateway:

    @pytest.fixture
    def csv_content(self):
        result = "systematic_name,standard_name,SM,SC,YPD\nYAL001C,FUN26,1.1,2.2,3.3\nYAL002W,AIM2,4.4,5.5,6.6\nYAL003W,BDH1,7.7,8.8,9.9\n"
        return result

    @pytest.fixture
    def file_path(self, tmp_path):
        result = tmp_path / "test.csv"
        return result

    @pytest.fixture
    def expected(self):
        result = [
            growth.GrowthRateData(
                growth_rate=2.2,
                ko_gene_systematic_name="YAL001C",
                ko_gene_standard_name="FUN26",
                medium=growth.GrowthMedium.SC,
            ),
            growth.GrowthRateData(
                growth_rate=5.5,
                ko_gene_systematic_name="YAL002W",
                ko_gene_standard_name="AIM2",
                medium=growth.GrowthMedium.SC,
            ),
            growth.GrowthRateData(
                growth_rate=8.8,
                ko_gene_systematic_name="YAL003W",
                ko_gene_standard_name="BDH1",
                medium=growth.GrowthMedium.SC,
            ),
        ]

        return result

    def test_get(self, csv_content, file_path, expected):
        file_path.write_text(csv_content)
        observed = growth_gateway.Yeast5kGrowthDataGateway(
            file_path=file_path, growth_medium=growth.GrowthMedium.SC
        ).get()
        assert observed == expected
