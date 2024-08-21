import pytest
import numpy as np
import pandas as pd

from bench.models.moma.preprocessing_utils import integrators


@pytest.fixture
def sample_omics_data():
    """Fixture to create sample OmicsData."""
    data = pd.DataFrame(
        {"Gene": ["Gene1", "Gene2", "Gene3"], "Expression": [10, 20, 30]}
    )
    return integrators.OmicsData(name="transcriptomics", data=data)


@pytest.fixture
def sample_multiomics_data(sample_omics_data):
    """Fixture to create sample MultiomicsData."""
    return integrators.MultiomicsData(
        proteomics=None, transcriptomics=sample_omics_data, fluxomics=None, growth=None
    )


def test_omics_data_initialization(sample_omics_data):
    """Test that OmicsData initializes correctly."""
    assert sample_omics_data.name == "transcriptomics"
    assert isinstance(sample_omics_data.data, pd.DataFrame)
    assert not sample_omics_data.data.empty


def test_omics_data_dataframe(sample_omics_data):
    """Test that the DataFrame in OmicsData is correctly stored and returned."""
    expected_data = pd.DataFrame(
        {"Gene": ["Gene1", "Gene2", "Gene3"], "Expression": [10, 20, 30]}
    )

    # Check that the actual DataFrame matches the expected one
    pd.testing.assert_frame_equal(sample_omics_data.data, expected_data)


def test_multiomics_data_to_dict(sample_multiomics_data, sample_omics_data):
    """Test that the to_dict method returns the correct DataFrame objects."""
    result_dict = sample_multiomics_data.to_dict()

    # Check that the result_dict contains the same DataFrame as in sample_omics_data
    assert "transcriptomics" in result_dict
    pd.testing.assert_frame_equal(
        result_dict["transcriptomics"], sample_omics_data.data
    )


def test_multiomics_data_to_dict_empty():
    """Test the to_dict method with no data."""
    multiomics_data = integrators.MultiomicsData()
    result_dict = multiomics_data.to_dict()

    # The result should be an empty dictionary
    assert result_dict == {}


class TestOmicsDataIntegrator:

    test_cases = [
        (
            "default_case",
            {
                "proteomics": pd.DataFrame(
                    {
                        "knockout_id": [
                            "ko1",
                            "ko2",
                            "ko2",
                            "ko3",
                            "ko4",
                            "ko5",
                            "ko6",
                        ],
                        "value": [10.5, 10.1, 20.7, 30.2, 40.5, 50.3, 60.8],
                    }
                ).set_index("knockout_id"),
                "transcriptomics": pd.DataFrame(
                    {
                        "knockout_id": [
                            "ko1",
                            "ko2",
                            "ko3",
                            "ko4",
                            "ko5",
                            "ko6",
                            "ko7",
                        ],
                        "value": [100, 200, 300, 400, 500, 600, 700],
                    }
                ).set_index("knockout_id"),
                "fluxomics": pd.DataFrame(
                    {
                        "knockout_id": ["ko1", "ko2", "ko3", "ko4", "ko5", "ko6"],
                        "value": [1000, 2000, 3000, 4000, 5000, 6000],
                    }
                ).set_index("knockout_id"),
            },
            {
                "proteomics": pd.DataFrame(
                    {
                        "knockout_id": ["ko1", "ko2", "ko3", "ko4", "ko5", "ko6"],
                        "value": [10.5, 10.1, 30.2, 40.5, 50.3, 60.8],
                    }
                ).set_index("knockout_id"),
                "transcriptomics": pd.DataFrame(
                    {
                        "knockout_id": ["ko1", "ko2", "ko3", "ko4", "ko5", "ko6"],
                        "value": [100, 200, 300, 400, 500, 600],
                    }
                ).set_index("knockout_id"),
                "fluxomics": pd.DataFrame(
                    {
                        "knockout_id": ["ko1", "ko2", "ko3", "ko4", "ko5", "ko6"],
                        "value": [1000, 2000, 3000, 4000, 5000, 6000],
                    }
                ).set_index("knockout_id"),
            },
        )
    ]

    @pytest.mark.parametrize(
        "case_id, input, expected_result",
        test_cases,
        ids=[case_id for case_id, _, _ in test_cases],
    )
    def test_integrate(self, case_id, input, expected_result):
        omics_data = {
            key: integrators.OmicsData(name=key, data=df) for key, df in input.items()
        }

        multi_omics_data = integrators.MultiomicsData(**omics_data)

        integrator = integrators.OmicsDataIntegrator()
        result = integrator.integrate(multiomics_data=multi_omics_data)

        for key in input.keys():
            # Check if indices are shuffled
            assert not np.array_equal(
                result[key].index, input[key].index
            ), f"Indices of {key} should be shuffled but are not."
            pd.testing.assert_frame_equal(
                result[key].sort_index(), expected_result[key].sort_index()
            )
