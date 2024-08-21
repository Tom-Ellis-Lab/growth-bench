import io
from unittest import mock

import pandas as pd
import pathlib
import pyreadr
import pytest

from bench.models.moma.gateways import gateways


# Sample DataFrame for testing
@pytest.fixture
def mock_dataframe():
    return pd.DataFrame(
        {
            "Column1": [1, 2, 3],
            "Column2": ["A", "B", "C"],
        }
    )


@pytest.fixture
def mock_duibhir_dataframe():
    return pd.DataFrame(
        {
            "Column1": [1, 2, 3],
            "Row": ["A", "B", "C"],
            "log2relT": [4, 5, 6],
        }
    )


# Define a mock function for pd.read_csv
def mock_read_csv(file_path):
    return pd.DataFrame(
        {
            "Column1": [1, 2, 3],
            "Column2": ["A", "B", "C"],
        }
    )


# Define a mock function for pyreadr.read_r
def mock_read_r(file_path):
    return {
        None: pd.DataFrame(
            {
                "Column1": [1, 2, 3],
                "Column2": ["A", "B", "C"],
            }
        )
    }


def test_proteomics_gateway(monkeypatch, mock_dataframe):
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    gateway = gateways.ProteomicsGateway(file_path=pathlib.Path("dummy_path.csv"))
    result = gateway.get()
    pd.testing.assert_frame_equal(result, mock_dataframe)


def test_yeast5k_growth_rates_gateway(monkeypatch, mock_dataframe):
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    gateway = gateways.Yeast5kGrowthRatesGateway(
        file_path=pathlib.Path("dummy_path.csv")
    )
    result = gateway.get()
    pd.testing.assert_frame_equal(result, mock_dataframe)


def test_culley_data_gateway(monkeypatch, mock_dataframe):
    monkeypatch.setattr(pyreadr, "read_r", mock_read_r)

    gateway = gateways.CulleyDataGateway(file_path=pathlib.Path("dummy_path.rds"))
    result = gateway.get()
    pd.testing.assert_frame_equal(result, mock_dataframe)


def test_transcriptomics_gateway(monkeypatch, mock_dataframe):
    monkeypatch.setattr(pyreadr, "read_r", mock_read_r)

    gateway = gateways.TranscriptomicsGateway(file_path=pathlib.Path("dummy_path.rds"))
    result = gateway.get()
    pd.testing.assert_frame_equal(result, mock_dataframe)


def test_fluxomics_gateway(mock_dataframe):

    transcriptomics_gateway = gateways.TranscriptomicsGateway(
        file_path=pathlib.Path("dummy_path.rds")
    )
    culley_gateway = gateways.CulleyDataGateway(
        file_path=pathlib.Path("dummy_path.rds")
    )

    # Set the return value of the get method
    transcriptomics_gateway.get = lambda: mock_dataframe
    culley_gateway.get = lambda: mock_dataframe

    gateway = gateways.FluxomicsGateway(
        transcriptomicsGateway=transcriptomics_gateway, culley_gateway=culley_gateway
    )
    result = gateway.get()

    # Here, adjust your expected result logic based on how FluxomicsGateway processes the data
    expected_result = mock_dataframe.drop(
        columns=mock_dataframe.columns
    )  # Adjust based on your logic
    pd.testing.assert_frame_equal(result, expected_result)


def test_duibhir_growth_rates_gateway(mock_duibhir_dataframe):

    culley_gateway = gateways.CulleyDataGateway(
        file_path=pathlib.Path("dummy_path.rds")
    )

    # Mock the get method
    culley_gateway.get = lambda: mock_duibhir_dataframe

    gateway = gateways.DuibhirGrowthRatesGateway(culley_gateway=culley_gateway)
    result = gateway.get()

    # Adjust this based on your actual data and expected output
    expected_result = mock_duibhir_dataframe[
        ["Row", "log2relT"]
    ]  # Replace with actual columns
    pd.testing.assert_frame_equal(result, expected_result)


class TestGeneNamesGateway:

    @pytest.fixture
    def mock_csv_data(self):
        return "col1\tcol2\nval1\tval2\nval3\tval4"

    def test_initialization(self):
        gateway = gateways.GeneNamesGateway(file_path="fake_path.tsv")
        assert gateway.file_path == "fake_path.tsv"

    def test_get(self, mock_csv_data):
        # Create the DataFrame directly from the mock CSV data
        mock_df = pd.read_csv(io.StringIO(mock_csv_data), sep="\t")

        # Mock the pandas.read_csv method to return the mock_df
        with mock.patch("pandas.read_csv", return_value=mock_df):
            gateway = gateways.GeneNamesGateway(file_path="fake_path.tsv")
            result = gateway.get()

            # Check that the DataFrame returned by the get method matches mock_df
            pd.testing.assert_frame_equal(result, mock_df)

    def test_get_systematic_name(self):
        gateway = gateways.GeneNamesGateway(file_path="fake_path.tsv")
        result = gateway.get_systematic_name()
        assert result == "systematic_name"

    def test_get_standard_name(self):
        gateway = gateways.GeneNamesGateway(file_path="fake_path.tsv")
        result = gateway.get_standard_name()
        assert result == "standard_name"
