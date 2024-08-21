import pytest
import pandas as pd

from bench.models.moma.preprocessing_utils import filters


@pytest.fixture
def filter_factory():
    return filters.FilterFactory()


class TestFilterFactory:

    def test_create_filter(self, filter_factory):
        filter_instance = filter_factory.create_filter(filters.FilterTypes.INTERSECTION)
        assert isinstance(filter_instance, filters.IntersectionFilter)

    def test_create_unknown_filter(self, filter_factory):
        with pytest.raises(ValueError):
            filter_factory.create_filter("unknown")  # Any invalid value


@pytest.fixture
def intersection_filter():
    return filters.IntersectionFilter()


class TestIntersectionFilter:

    def test_filter_data(self, intersection_filter):
        # Prepare sample data
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        df2 = pd.DataFrame({"B": [4, 5, 6]}, index=[2, 3, 4])
        data = {"df1": df1, "df2": df2}

        # Perform filtering
        filtered_data = intersection_filter.filter_data(data)

        # Expected result should only contain index 2 and 3
        expected_df1 = pd.DataFrame({"A": [2, 3]}, index=[2, 3])
        expected_df2 = pd.DataFrame({"B": [4, 5]}, index=[2, 3])
        expected_result = {"df1": expected_df1, "df2": expected_df2}

        pd.testing.assert_frame_equal(filtered_data["df1"], expected_result["df1"])
        pd.testing.assert_frame_equal(filtered_data["df2"], expected_result["df2"])

    def test_filter_data_with_insufficient_datasets(self, intersection_filter):
        # Prepare a single dataframe
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        data = {"df1": df1}

        # Should raise an error as at least two datasets are required
        with pytest.raises(ValueError):
            intersection_filter.filter_data(data)

    def test_filter_data_with_no_common_indices(self, intersection_filter):
        # Prepare sample data with no common indices
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[1, 2, 3])
        df2 = pd.DataFrame({"B": [4, 5, 6]}, index=[4, 5, 6])
        data = {"df1": df1, "df2": df2}

        # Perform filtering
        filtered_data = intersection_filter.filter_data(data)

        assert len(filtered_data) == 2
        assert len(filtered_data["df1"]) == 0
        assert len(filtered_data["df2"]) == 0
