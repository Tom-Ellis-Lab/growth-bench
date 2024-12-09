import numpy as np
import pytest
import pandas as pd
from bench.models.moma import normalisers
from bench.models.moma.preprocessing_utils import integrators


class TestNormalisationParam:

    @pytest.fixture
    def input(self):
        x_train = [
            integrators.OmicsData(
                name="proteomics", data=pd.DataFrame([[1, 2], [3, 4]])
            ),
            integrators.OmicsData(
                name="transcriptomics", data=pd.DataFrame([[5, 6], [7, 8]])
            ),
        ]

        x_val = [
            integrators.OmicsData(
                name="proteomics", data=pd.DataFrame([[2, 3], [4, 5]])
            ),
            integrators.OmicsData(
                name="transcriptomics", data=pd.DataFrame([[6, 7], [8, 9]])
            ),
        ]

        result = normalisers.NormalisationParam(x_train=x_train, x_val=x_val)

        return result

    def test_normalisation_param_valid(self, input):
        params = normalisers.NormalisationParam(
            x_train=input.x_train, x_val=input.x_val
        )
        assert params.x_train == input.x_train
        assert params.x_val == input.x_val


class TestStandardScalerNormaliser:

    @pytest.fixture
    def input(self):
        x_train = [
            integrators.OmicsData(
                name="proteomics", data=pd.DataFrame([[1, 2], [3, 4]])
            ),
            integrators.OmicsData(
                name="transcriptomics", data=pd.DataFrame([[5, 6], [7, 8]])
            ),
        ]

        x_val = [
            integrators.OmicsData(
                name="proteomics", data=pd.DataFrame([[2, 3], [4, 5]])
            ),
            integrators.OmicsData(
                name="transcriptomics", data=pd.DataFrame([[6, 7], [8, 9]])
            ),
        ]

        result = normalisers.NormalisationParam(x_train=x_train, x_val=x_val)

        return result

    @pytest.fixture
    def expected(self):
        scaled_x_train = [
            normalisers.ScaledData(
                name="proteomics", data=np.array([[-1.0, -1.0], [1.0, 1.0]])
            ),
            normalisers.ScaledData(
                name="transcriptomics", data=np.array([[-1.0, -1.0], [1.0, 1.0]])
            ),
        ]

        scaled_x_val = [
            normalisers.ScaledData(
                name="proteomics", data=np.array([[0.0, 0.0], [2.0, 2.0]])
            ),
            normalisers.ScaledData(
                name="transcriptomics", data=np.array([[0.0, 0.0], [2.0, 2.0]])
            ),
        ]

        result = normalisers.NormalisedDataset(
            scaled_x_train=scaled_x_train, scaled_x_val=scaled_x_val
        )
        return result

    def test_normalise(
        self,
        input,
        expected,
    ):
        normaliser = normalisers.StandardScalerNormaliser()

        observed = normaliser.normalise(input)

        assert isinstance(observed, normalisers.NormalisedDataset)
        for exp_data, obs_data in zip(expected.scaled_x_train, observed.scaled_x_train):
            assert exp_data.name == obs_data.name
            assert np.array_equal(exp_data.data, obs_data.data)

        for exp_data, obs_data in zip(expected.scaled_x_val, observed.scaled_x_val):
            assert exp_data.name == obs_data.name
            assert np.array_equal(exp_data.data, obs_data.data)
