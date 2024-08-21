import pandas as pd
import pytest

from bench.models.moma import data_splitters


class TestDataSplitter:

    @pytest.fixture
    def data(self):
        result = pd.DataFrame(
            {
                "knockout_id": [1, 2, 3, 4, 5],
                "growth_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        return result

    @pytest.fixture
    def splitter(self):
        result = data_splitters.DataSplitter()
        return result

    def test_split(self, data, splitter):
        result = splitter.split(data, test_size=0.2)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) == 4
        assert len(result["test"]) == 1
        assert result["train"].shape[0] == 4
        assert result["test"].shape[0] == 1
        assert result["train"].shape[1] == 2
        assert result["test"].shape[1] == 2

    @pytest.fixture
    def multiple_data(self):
        result = {
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
        return result

    def test_split_multiple_data(self, multiple_data, splitter):
        result = splitter.split_multiple_data(multiple_data, test_size=0.2)
        assert "growth" in result["train"]
        assert "growth" in result["test"]
        assert "proteomics" in result["train"]
        assert "proteomics" in result["test"]
        assert len(result["train"]["growth"]) == 4
        assert len(result["test"]["growth"]) == 1
        assert len(result["train"]["proteomics"]) == 4
        assert len(result["test"]["proteomics"]) == 1
        assert result["train"]["growth"].shape[0] == 4
        assert result["test"]["growth"].shape[0] == 1
        assert result["train"]["proteomics"].shape[0] == 4
        assert result["test"]["proteomics"].shape[0] == 1
        assert result["train"]["growth"].shape[1] == 2
        assert result["test"]["growth"].shape[1] == 2
        assert result["train"]["proteomics"].shape[1] == 2
        assert result["test"]["proteomics"].shape[1] == 2

    @pytest.fixture
    def data_for_cv(self):
        result = pd.DataFrame(
            {
                "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "growth_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            }
        )
        return result

    def test_split_for_cross_validation(self, data_for_cv, splitter):
        result = splitter.split_for_cross_validation(data_for_cv, cross_validation=5)
        assert len(result) == 5

        # check the keys
        folds = [key for key in result.keys()]
        assert folds == [1, 2, 3, 4, 5]

        # do a loop to check the length of each fold
        for key in result.keys():
            assert len(result[key]) == 2  # 10 divided by 5 folds is 2
            assert len(result[key]["train"]) == 8  # Size of training set: 10 - 2 = 8
            assert len(result[key]["test"]) == 2  # Size of testing set: 2
            assert result[key]["train"].shape[0] == 8  # 8 rows in training set
            assert result[key]["test"].shape[0] == 2  # 2 rows in testing set
            assert result[key]["train"].shape[1] == 2  # 2 columns in training set
            assert result[key]["test"].shape[1] == 2  # 2 columns in testing set
            assert "knockout_id" in result[key]["train"]
            assert "knockout_id" in result[key]["test"]
            assert "growth_rate" in result[key]["train"]
            assert "growth_rate" in result[key]["test"]

    @pytest.fixture
    def multiple_data_for_cv(self):
        result = {
            "growth": pd.DataFrame(
                {
                    "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "growth_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                }
            ),
            "proteomics": pd.DataFrame(
                {
                    "knockout_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "protein_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }
            ),
        }

        return result

    def test_split_multiple_data_for_cross_validation(
        self, multiple_data_for_cv, splitter
    ):
        result = splitter.split_multiple_data_for_cross_validation(
            multiple_data_for_cv, cross_validation=5
        )
        assert len(result) == 5

        # check the keys
        folds = [key for key in result.keys()]
        assert folds == [1, 2, 3, 4, 5]
        # do a loop to check the length of each fold
        for key in result.keys():
            assert len(result[key]) == 2
            assert len(result[key]["train"]["growth"]) == 8
            assert len(result[key]["test"]["growth"]) == 2
            assert len(result[key]["train"]["proteomics"]) == 8
            assert len(result[key]["test"]["proteomics"]) == 2
            assert result[key]["train"]["growth"].shape[0] == 8
            assert result[key]["test"]["growth"].shape[0] == 2
            assert result[key]["train"]["proteomics"].shape[0] == 8
            assert result[key]["test"]["proteomics"].shape[0] == 2
            assert result[key]["train"]["growth"].shape[1] == 2
            assert result[key]["test"]["growth"].shape[1] == 2
            assert result[key]["train"]["proteomics"].shape[1] == 2
            assert result[key]["test"]["proteomics"].shape[1] == 2
            assert "knockout_id" in result[key]["train"]["growth"]
            assert "knockout_id" in result[key]["test"]["growth"]
            assert "growth_rate" in result[key]["train"]["growth"]
            assert "growth_rate" in result[key]["test"]["growth"]
