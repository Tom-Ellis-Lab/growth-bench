import numpy as np
import pytest
import pandas as pd
from bench.models.moma import normalisers


@pytest.fixture
def sample_data():
    return {
        "train": {
            "feature1": pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"]),
            "feature2": pd.DataFrame([[5, 6], [7, 8]], columns=["C", "D"]),
            "target": pd.DataFrame([1, 0], columns=["Label"]),
        },
        "test": {
            "feature1": pd.DataFrame([[2, 3], [4, 5]], columns=["A", "B"]),
            "feature2": pd.DataFrame([[6, 7], [8, 9]], columns=["C", "D"]),
            "target": pd.DataFrame([0, 1], columns=["Label"]),
        },
    }


class TestNormalisationParam:

    def test_normalisation_param_valid(self, sample_data):
        params = normalisers.NormalisationParam(data=sample_data, target_name="target")
        assert params.data == sample_data

    def test_normalisation_param_type_checks(self, sample_data):

        # Test that values in the outer dictionary are dictionaries
        invalid_data = {"train": "this is not a dict", "test": sample_data["test"]}
        with pytest.raises(ValueError, match="The values must be dictionaries."):
            normalisers.NormalisationParam(data=invalid_data, target_name="target")

        # Test that keys in the inner dictionary are strings
        invalid_data = {
            "train": {
                0: pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"]),
                "target": pd.DataFrame([1, 0], columns=["Label"]),
            },
            "test": sample_data["test"],
        }
        with pytest.raises(ValueError, match="The keys must be strings."):
            normalisers.NormalisationParam(data=invalid_data, target_name="target")

        # Test that values in the inner dictionary are pandas DataFrames
        invalid_data = {
            "train": {
                "feature1": "this is not a DataFrame",
                "target": pd.DataFrame([1, 0], columns=["Label"]),
            },
            "test": sample_data["test"],
        }
        with pytest.raises(ValueError, match="The values must be pandas DataFrames."):
            normalisers.NormalisationParam(data=invalid_data, target_name="target")

    def test_normalisation_param_missing_target(self, sample_data):
        invalid_data = sample_data.copy()
        invalid_data["train"].pop("target")

        with pytest.raises(ValueError, match="Data must contain the target: target."):
            normalisers.NormalisationParam(data=invalid_data, target_name="target")

        # Test missing "train" key
        invalid_data = sample_data.copy()
        invalid_data.pop("train")

        with pytest.raises(
            ValueError, match="Data must have exactly two keys: 'train' and 'test'."
        ):
            normalisers.NormalisationParam(data=invalid_data, target_name="target")

        # Test missing "test" key
        invalid_data = sample_data.copy()
        invalid_data.pop("test")

        with pytest.raises(
            ValueError, match="Data must have exactly two keys: 'train' and 'test'."
        ):
            normalisers.NormalisationParam(data=invalid_data, target_name="target")


class TestStandardScalerNormaliser:

    @pytest.fixture
    def expected_data(self):
        return {
            "scaled_train": {
                "feature1": np.array([[-1.0, -1.0], [1.0, 1.0]]),
                "feature2": np.array([[-1.0, -1.0], [1.0, 1.0]]),
                "target": pd.DataFrame([1, 0], columns=["Label"]),
            },
            "scaled_test": {
                "feature1": np.array([[0.0, 0.0], [2.0, 2.0]]),
                "feature2": np.array([[0.0, 0.0], [2.0, 2.0]]),
                "target": pd.DataFrame([0, 1], columns=["Label"]),
            },
            "train": {
                "feature1": pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"]),
                "feature2": pd.DataFrame([[5, 6], [7, 8]], columns=["C", "D"]),
                "target": pd.DataFrame([1, 0], columns=["Label"]),
            },
            "test": {
                "feature1": pd.DataFrame([[2, 3], [4, 5]], columns=["A", "B"]),
                "feature2": pd.DataFrame([[6, 7], [8, 9]], columns=["C", "D"]),
                "target": pd.DataFrame([0, 1], columns=["Label"]),
            },
        }

    def test_normalise(self, sample_data, expected_data):
        normaliser = normalisers.StandardScalerNormaliser()
        params = normalisers.NormalisationParam(data=sample_data, target_name="target")

        scaled_data = normaliser.normalise(params)

        print(scaled_data)

        # Check if scaled_train and scaled_test keys are present
        assert "scaled_train" in scaled_data
        assert "scaled_test" in scaled_data

        # Validate scaling
        for key, value in sample_data["train"].items():
            if key != "target":
                pd.testing.assert_frame_equal(
                    pd.DataFrame(
                        scaled_data["scaled_train"][key],
                        columns=value.columns,
                        index=value.index,
                    ),
                    pd.DataFrame(
                        expected_data["scaled_train"][key],
                        columns=value.columns,
                        index=value.index,
                    ),
                )

                pd.testing.assert_frame_equal(
                    pd.DataFrame(
                        scaled_data["scaled_test"][key],
                        columns=sample_data["test"][key].columns,
                        index=sample_data["test"][key].index,
                    ),
                    pd.DataFrame(
                        expected_data["scaled_test"][key],
                        columns=sample_data["test"][key].columns,
                        index=sample_data["test"][key].index,
                    ),
                )
