import pandas as pd
import pytest

from bench.models.moma import data_splitters
from bench.models.moma.preprocessing_utils import integrators


class TestDataSplitterParams:

    @pytest.fixture
    def valid_data(self):
        """Fixture for creating valid MultiomicsData."""
        df1 = pd.DataFrame(
            {
                "value": [1, 2, 3],
            },
            index=["a", "b", "c"],
        )

        df2 = pd.DataFrame(
            {
                "value": [10, 20, 30],
            },
            index=["a", "b", "c"],
        )

        return integrators.MultiomicsData(
            proteomics=integrators.OmicsData(name="proteomics", data=df1),
            transcriptomics=integrators.OmicsData(name="transcriptomics", data=df2),
        )

    @pytest.fixture
    def invalid_data(self):
        """Fixture for creating invalid MultiomicsData."""
        df1 = pd.DataFrame(
            {
                "value": [1, 2, 3],
            },
            index=["a", "b", "c"],
        )

        df2 = pd.DataFrame(
            {
                "value": [10, 20, 30],
            },
            index=["a", "b", "d"],
        )  # Different index

        return integrators.MultiomicsData(
            proteomics=integrators.OmicsData(name="proteomics", data=df1),
            transcriptomics=integrators.OmicsData(name="transcriptomics", data=df2),
        )

    def test_valid_indices(self, valid_data):
        """Test that DataSplitterParams does not raise an exception with valid indices."""
        try:
            params = data_splitters.DataSplitterParams(data=valid_data)
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError raised: {e}")

    def test_invalid_indices(self, invalid_data):
        """Test that DataSplitterParams raises a ValueError with invalid indices."""
        with pytest.raises(
            ValueError,
            match="Dataframes must have the same indices. Mismatch found in dataframe with key: 'transcriptomics'.",
        ):
            data_splitters.DataSplitterParams(data=invalid_data)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "id": "single_omics_data",
            "input": {
                "growth": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5],
                        "growth_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
                    }
                ),
                "proteomics": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5],
                        "protein_id": [1, 2, 3, 4, 5],
                    }
                ),
            },
            "expected": {"train_size": 4, "test_size": 1, "num_features": 2},
        },
        {
            "id": "multiple_omics_data",
            "input": {
                "growth": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6],
                        "growth_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    }
                ),
                "proteomics": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6],
                        "protein_id": [1, 2, 3, 4, 5, 6],
                    }
                ),
                "transcriptomics": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6],
                        "transcript_id": [1, 2, 3, 4, 5, 6],
                    }
                ),
            },
            "expected": {"train_size": 4, "test_size": 2, "num_features": 2},
        },
    ],
    ids=lambda case: case["id"],
)
class TestDataSplitter:

    @pytest.fixture
    def splitter(self):
        """Fixture for creating a DataSplitter instance."""
        return data_splitters.DataSplitter()

    @pytest.fixture
    def params(self, test_case):
        """Fixture for creating DataSplitterParams with parameterized data."""
        # Convert test_case input to MultiomicsData
        data_dict = test_case["input"]
        omics_data_dict = {
            key: integrators.OmicsData(name=key, data=df)
            for key, df in data_dict.items()
        }
        multiomics_data = integrators.MultiomicsData(**omics_data_dict)

        # Return DataSplitterParams
        return data_splitters.DataSplitterParams(
            data=multiomics_data, test_size=0.2, random_state=42, shuffle=True
        )

    def test_split(self, splitter, params, test_case):
        observed = splitter.split(params)

        # Assertions for train and test sets
        for key in test_case["input"].keys():
            assert key in observed["train"]
            assert key in observed["test"]

            train_size = test_case["expected"]["train_size"]
            test_size = test_case["expected"]["test_size"]
            num_features = test_case["expected"]["num_features"]

            # Validate the sizes of train and test sets
            assert len(observed["train"][key]) == train_size
            assert len(observed["test"][key]) == test_size

            # Validate the shape (rows and columns) of the dataframes
            assert observed["train"][key].shape[0] == train_size
            assert observed["test"][key].shape[0] == test_size
            assert observed["train"][key].shape[1] == num_features
            assert observed["test"][key].shape[1] == num_features


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "id": "single_omics_data",
            "data": {
                "growth": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "growth_rate": [
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            0.7,
                            0.8,
                            0.9,
                            1.0,
                        ],
                    }
                ),
                "proteomics": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "protein_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
                ),
            },
            "expected": {
                "num_folds": 5,
                "train_size": 8,
                "test_size": 2,
                "num_features": 2,
            },
        },
        {
            "id": "multiple_omics_data",
            "data": {
                "growth": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "growth_rate": [
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            0.7,
                            0.8,
                            0.9,
                            1.0,
                        ],
                    }
                ),
                "proteomics": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "protein_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
                ),
                "transcriptomics": pd.DataFrame(
                    {
                        "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "transcript_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    }
                ),
            },
            "expected": {
                "num_folds": 5,
                "train_size": 8,
                "test_size": 2,
                "num_features": 2,
            },
        },
    ],
    ids=lambda case: case["id"],
)
class TestCrossValidationDataSplitter:

    @pytest.fixture
    def params(self, test_case):
        """Fixture for creating DataSplitterParams with parameterized data."""
        # Convert test_case data to MultiomicsData
        omics_data = {
            key: integrators.OmicsData(name=key, data=df)
            for key, df in test_case["data"].items()
        }
        multiomics_data = integrators.MultiomicsData(**omics_data)

        # Return DataSplitterParams
        return data_splitters.DataSplitterParams(
            data=multiomics_data,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            cross_validation=test_case["expected"]["num_folds"],
        )

    def test_split(self, params, test_case):
        """Test the cross-validation splitting functionality of the CrossValidationDataSplitter."""
        splitter = data_splitters.CrossValidationDataSplitter()
        result = splitter.split(params)

        # Check the number of folds
        expected_num_folds = test_case["expected"]["num_folds"]
        assert len(result) == expected_num_folds

        # Check the keys
        folds = [key for key in result.keys()]
        expected_folds = list(range(1, expected_num_folds + 1))
        assert folds == expected_folds

        # Check the content of each fold
        for key in result.keys():
            assert len(result[key]) == 2  # "train" and "test"
            assert (
                len(result[key]["train"]["growth"])
                == test_case["expected"]["train_size"]
            )
            assert (
                len(result[key]["test"]["growth"]) == test_case["expected"]["test_size"]
            )
            assert (
                len(result[key]["train"]["proteomics"])
                == test_case["expected"]["train_size"]
            )
            assert (
                len(result[key]["test"]["proteomics"])
                == test_case["expected"]["test_size"]
            )

            assert (
                result[key]["train"]["growth"].shape[0]
                == test_case["expected"]["train_size"]
            )
            assert (
                result[key]["test"]["growth"].shape[0]
                == test_case["expected"]["test_size"]
            )
            assert (
                result[key]["train"]["proteomics"].shape[0]
                == test_case["expected"]["train_size"]
            )
            assert (
                result[key]["test"]["proteomics"].shape[0]
                == test_case["expected"]["test_size"]
            )

            assert (
                result[key]["train"]["growth"].shape[1]
                == test_case["expected"]["num_features"]
            )
            assert (
                result[key]["test"]["growth"].shape[1]
                == test_case["expected"]["num_features"]
            )
            assert (
                result[key]["train"]["proteomics"].shape[1]
                == test_case["expected"]["num_features"]
            )
            assert (
                result[key]["test"]["proteomics"].shape[1]
                == test_case["expected"]["num_features"]
            )

            assert "knockout_id" in result[key]["train"]["growth"]
            assert "knockout_id" in result[key]["test"]["growth"]
            assert "growth_rate" in result[key]["train"]["growth"]
            assert "growth_rate" in result[key]["test"]["growth"]


class TestCrossValidationSplitterError:

    @pytest.fixture
    def data(self):
        data = {
            "growth": pd.DataFrame(
                {
                    "knockout_id": [1, 2, 3, 4, 5],
                    "growth_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
                }
            ),
            "proteomics": pd.DataFrame(
                {
                    "knockout_id": [1, 2, 3, 4, 5],
                    "protein_id": [1, 2, 3, 4, 5],
                }
            ),
        }

        omics_data = {
            key: integrators.OmicsData(name=key, data=df) for key, df in data.items()
        }

        result = integrators.MultiomicsData(**omics_data)
        return result

    @pytest.fixture
    def wrong_params(self, data):
        """Fixture for creating DataSplitterParams with missing cross-validation parameter."""
        return data_splitters.DataSplitterParams(
            data=data, test_size=0.2, random_state=42, shuffle=True
        )

    def test_wrong_params(self, wrong_params):
        """Test that missing cross-validation parameter raises an exception."""
        splitter = data_splitters.CrossValidationDataSplitter()
        with pytest.raises(
            ValueError, match="Cross-validation parameter must be provided."
        ):
            splitter.split(wrong_params)
