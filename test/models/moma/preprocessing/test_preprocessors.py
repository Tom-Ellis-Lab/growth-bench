import pytest
from unittest import mock
import pandas as pd

from bench.models.moma.gateways import gateways
from bench.models.moma.preprocessing_utils import preprocessors


@pytest.fixture
def mock_gene_names_mapper():
    return mock.Mock(spec=gateways.GeneNamesGateway)


class TestProteomicsPreprocessor:
    data = [
        (
            "default_case",
            pd.DataFrame(
                {
                    "col1": ["gene_A", "gene_B", "gene_C"],
                    "gene_ko_a_ko_b": [4, 5, 6],
                    "gene_ko_c": [7, 8, 9],
                    "_qc_qc_qc_1": [0, 0, 0],
                    "_HIS3_control": [10, 11, 12],
                }
            ),
            pd.DataFrame(
                {
                    "protein_id": ["gene_A", "gene_B", "gene_C"],
                    "a": [4, 5, 6],
                    "c": [7, 8, 9],
                }
            )
            .set_index("protein_id")
            .T,
        ),
        (
            "no_quality_controls",
            pd.DataFrame(
                {
                    "col1": ["gene_A", "gene_B", "gene_C"],
                    "gene_ko_a_ko_b": [4, 5, 6],
                    "gene_ko_c": [7, 8, 9],
                }
            ),
            pd.DataFrame(
                {
                    "protein_id": ["gene_A", "gene_B", "gene_C"],
                    "a": [4, 5, 6],
                    "c": [7, 8, 9],
                }
            )
            .set_index("protein_id")
            .T,
        ),
        (
            "no_his3_controls",
            pd.DataFrame(
                {
                    "col1": ["gene_A", "gene_B", "gene_C"],
                    "gene_ko_a_ko_b": [4, 5, 6],
                    "gene_ko_c": [7, 8, 9],
                    "_qc_qc_qc_1": [0, 0, 0],
                }
            ),
            pd.DataFrame(
                {
                    "protein_id": ["gene_A", "gene_B", "gene_C"],
                    "a": [4, 5, 6],
                    "c": [7, 8, 9],
                }
            )
            .set_index("protein_id")
            .T,
        ),
        (
            "no_controls",
            pd.DataFrame(
                {
                    "col1": ["gene_A", "gene_B", "gene_C"],
                    "gene_ko_a_ko_b": [4, 5, 6],
                    "gene_ko_c": [7, 8, 9],
                }
            ),
            pd.DataFrame(
                {
                    "protein_id": ["gene_A", "gene_B", "gene_C"],
                    "a": [4, 5, 6],
                    "c": [7, 8, 9],
                }
            )
            .set_index("protein_id")
            .T,
        ),
    ]

    @pytest.mark.parametrize(
        "id_, data, expected", data, ids=[id_ for id_, _, _ in data]
    )
    def test_preprocess(self, id_, data, expected):
        gateway = mock.Mock(spec=gateways.ProteomicsGateway)
        gateway.get = mock.Mock(return_value=data)

        preprocessor = preprocessors.ProteomicsPreprocessor(gateway=gateway)
        result = preprocessor.preprocess()

        pd.testing.assert_frame_equal(result, expected)


class TestYeast5kGrowthRatesPreprocessor:
    data = [
        (
            "default_case",
            preprocessors.GrowthMedium.SC,
            pd.DataFrame(
                {
                    "orf": ["gene_A", "gene_B", "gene_C"],
                    "SC": [0.1, 0.2, 0.3],
                    "SM": [0.4, 0.5, 0.6],
                    "YPD": [0.7, 0.8, 0.9],
                    "SD": [1.0, 1.1, 1.2],
                }
            ),
            pd.DataFrame(
                {
                    "growth_rate_SC": [0.1, 0.2, 0.3],
                    "systematic_name": ["gene_A", "gene_B", "gene_C"],
                }
            ).set_index("systematic_name"),
        ),
        # Add more test cases as needed
    ]

    @pytest.mark.parametrize(
        "id_, growth_medium, data, expected",
        data,
        ids=[id_ for id_, _, _, _ in data],
    )
    def test_preprocess(self, id_, growth_medium, data, expected):
        gateway = mock.Mock(spec=preprocessors.Yeast5kGrowthRatesPreprocessor)
        gateway.get = mock.Mock(return_value=data)
        preprocessor = preprocessors.Yeast5kGrowthRatesPreprocessor(gateway=gateway)
        result = preprocessor.preprocess(growth_medium=growth_medium)
        pd.testing.assert_frame_equal(result, expected)


class TestTranscriptomicsPreprocessor:

    test_cases = [
        (
            "default_case",
            pd.DataFrame(
                {
                    "log2relT": [0.5, 1.5, 2.5],
                    "YMR056C": [10, 10, 20],
                    "YBR085W": [0, 0, 20],
                    "YER033C": [0, 0, 0],
                }
            ),
            pd.DataFrame(
                {
                    "knockout_id": ["YAL009W", "YAL011W", "YAL013W"],
                    "YMR056C": [10, 10, 20],
                    "YBR085W": [0, 0, 20],
                },
            ).set_index("knockout_id"),
        ),
        # Add more test cases if neededs
    ]

    @pytest.fixture
    def mock_transcriptomics_gateway(self):
        return mock.Mock(spec=gateways.TranscriptomicsGateway)

    @pytest.fixture
    def mock_culley_gateway(self):
        return mock.Mock(spec=gateways.CulleyDataGateway)

    @pytest.mark.parametrize(
        "id_, data, expected", test_cases, ids=[id_ for id_, _, _ in test_cases]
    )
    def test_preprocess(
        self,
        id_,
        data,
        expected,
        mock_transcriptomics_gateway,
        mock_culley_gateway,
        mock_gene_names_mapper,
    ):
        mock_transcriptomics_gateway.get = mock.Mock(return_value=data)
        mock_culley_gateway.get = mock.Mock(
            return_value=pd.DataFrame({"Row": ["SPO7", "SWC3", "DEP1"]})
        )
        mock_gene_names_mapper.map = mock.Mock(
            return_value=data.assign(knockout_id=["YAL009W", "YAL011W", "YAL013W"])
        )
        preprocessor = preprocessors.TranscriptomicsPreprocessor(
            transcriptomics_gateway=mock_transcriptomics_gateway,
            culley_gateway=mock_culley_gateway,
            gene_names_mapper=mock_gene_names_mapper,
        )
        result = preprocessor.preprocess()
        pd.testing.assert_frame_equal(result, expected)


class TestFluxomicsPreprocessor:

    data = [
        (
            "default_case",
            pd.DataFrame(
                {
                    "Row": ["SPO7", "SWC3", "DEP1"],
                    "r_0001": [10, 10, 20],
                    "r_0002": [0, 0, 20],
                    "r_0003": [0, 0, 0],
                }
            ),
            pd.DataFrame(
                {
                    "knockout_id": ["YAL009W", "YAL011W", "YAL013W"],
                    "r_0001": [10, 10, 20],
                    "r_0002": [0, 0, 20],
                },
            ).set_index("knockout_id"),
        ),
        # Add more test cases if neededs
    ]

    @pytest.fixture
    def mock_fluxomics_gateway(self):
        return mock.Mock(spec=gateways.FluxomicsGateway)

    @pytest.mark.parametrize(
        "id_, input, expected", data, ids=[id_ for id_, _, _ in data]
    )
    def test_preprocess(
        self, id_, input, expected, mock_fluxomics_gateway, mock_gene_names_mapper
    ):
        mock_fluxomics_gateway.get = mock.Mock(return_value=input)

        # Mock the map method to correctly simulate gene name mapping
        def mock_map(data, target_col, new_col, fillna_col):
            mapping_dict = {"SPO7": "YAL009W", "SWC3": "YAL011W", "DEP1": "YAL013W"}
            data[new_col] = data[target_col].map(mapping_dict).fillna(data[fillna_col])
            return data

        mock_gene_names_mapper.map = mock.Mock(side_effect=mock_map)

        preprocessor = preprocessors.FluxomicsPreprocessor(
            fluxomics_gateway=mock_fluxomics_gateway,
            gene_names_mapper=mock_gene_names_mapper,
        )
        result = preprocessor.preprocess()

        pd.testing.assert_frame_equal(result, expected)


class TestDuibhirGrowthRatesPreprocessor:

    data = [
        (
            "default_case",
            pd.DataFrame(
                {
                    "Row": ["SPO7", "SWC3", "DEP1"],
                    "log2relT": [0.5, 1.5, 2.5],
                }
            ),
            pd.DataFrame(
                {
                    "knockout_id": ["YAL009W", "YAL011W", "YAL013W"],
                    "growth_rate": [0.5, 1.5, 2.5],
                },
            ).set_index("knockout_id"),
        ),
        # Add more test cases if neededs
    ]

    @pytest.fixture
    def mock_growth_rates_gateway(self):
        return mock.Mock(spec=gateways.DuibhirGrowthRatesGateway)

    @pytest.mark.parametrize(
        "id_, input, expected", data, ids=[id_ for id_, _, _ in data]
    )
    def test_preprocess(
        self, id_, input, expected, mock_growth_rates_gateway, mock_gene_names_mapper
    ):
        mock_growth_rates_gateway.get = mock.Mock(return_value=input)

        # Mock the map method to correctly simulate gene name mapping
        def mock_map(data, target_col, new_col, fillna_col):
            mapping_dict = {"SPO7": "YAL009W", "SWC3": "YAL011W", "DEP1": "YAL013W"}
            data[new_col] = data[target_col].map(mapping_dict).fillna(data[fillna_col])
            return data

        mock_gene_names_mapper.map = mock.Mock(side_effect=mock_map)

        preprocessor = preprocessors.DuibhirGrowthRatesPreprocessor(
            growth_rates_gateway=mock_growth_rates_gateway,
            gene_names_mapper=mock_gene_names_mapper,
        )
        result = preprocessor.preprocess()

        pd.testing.assert_frame_equal(result, expected)
