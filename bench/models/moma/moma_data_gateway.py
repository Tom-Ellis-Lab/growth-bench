import csv
import dataclasses

import pyreadr
import numpy as np
import pandas as pd

from sklearn import preprocessing

TARGET_NAME = "log2relT"  # Based on the data for moma

"""
Note:
This gateway is responsible for loading the MOMA data from the files in and preprocessing it:
- loading the gene expression data
- loading the flux data
- loading the target data
- splitting the data into training and testing data
- scaling the data

Use this gateway as an interface for different tasks across the growth-bench repository.
"""


@dataclasses.dataclass
class MomaDataclass:
    """Dataclass for MOMA data."""

    scaled_gene_expression_training_data: np.ndarray
    scaled_gene_expression_testing_data: np.ndarray
    scaled_flux_training_data: np.ndarray
    scaled_flux_testing_data: np.ndarray
    target_training_data: np.ndarray
    target_testing_data: np.ndarray


def get_gene_expression_training_data(
    gene_expression_data: pd.DataFrame, testing_data_indices: list[int]
) -> pd.DataFrame:
    """Get the gene expression training data.

    Parameters:
    -----------
    gene_expression_data: pd.DataFrame
        The gene expression data.
    testing_data_indices: list[int]
        The indices of the testing data.

    Returns:
    --------
    pd.DataFrame
        The gene expression training data.
    """
    result = gene_expression_data.drop(gene_expression_data.index[testing_data_indices])

    return result


def get_gene_expression_testing_data(
    gene_expression_data: pd.DataFrame, testing_data_indices: list[int]
) -> pd.DataFrame:
    """Get the gene expression testing data.

    Parameters:
    -----------
    gene_expression_data: pd.DataFrame
        The gene expression data.
    testing_data_indices: list[int]
        The indices of the testing data.

    Returns:
    --------
    pd.DataFrame
        The gene expression testing data.
    """
    result = gene_expression_data.iloc[testing_data_indices, :]

    return result


def get_flux_training_data(
    flux_data: pd.DataFrame, testing_data_indices: list[int]
) -> pd.DataFrame:
    """Get the flux training data.

    Parameters:
    -----------
    flux_data: pd.DataFrame
        The flux data.
    testing_data_indices: list[int]
        The indices of the testing data.

    Returns:
    --------
    pd.DataFrame
        The flux training data.
    """
    result = flux_data.drop(flux_data.index[testing_data_indices])

    return result


def get_flux_testing_data(
    flux_data: pd.DataFrame, testing_data_indices: list[int]
) -> pd.DataFrame:
    """Get the flux testing data.

    Parameters:
    -----------
    flux_data: pd.DataFrame
        The flux data.
    testing_data_indices: list[int]
        The indices of the testing data.

    Returns:
    --------
    pd.DataFrame
        The flux testing data.
    """
    result = flux_data.iloc[testing_data_indices, :]

    return result


def get_gene_expression_data(
    file_path: str, target_name: str = TARGET_NAME
) -> pd.DataFrame:
    """Get the gene expression data.

    Parameters:
    -----------
    file_path: str
        The path to the file containing the gene expression data.
    target_name: str (optional)
        The name of the target data.

    Returns:
    --------
    pd.DataFrame
        The gene expression data.

    """
    gene_expression_data = pyreadr.read_r(file_path)[None]
    if gene_expression_data is None:
        print("error with gene expression data, not able to read")
    else:
        print("Success in loading gene expression data")
        print("Shape of the gene expression data:", gene_expression_data.shape)
    # Remove columns consisting of only zeros
    gene_expression_data = gene_expression_data.loc[
        :, (gene_expression_data != 0).any(axis=0)
    ]
    # Drop target data from gene expression data
    result = gene_expression_data.drop(columns=[target_name])

    return result


def get_flux_data(
    complete_data: pd.DataFrame,
    gene_expression_data: pd.DataFrame,
    target_name: str = TARGET_NAME,
) -> pd.DataFrame:
    """Get the flux data.

    Parameters:
    -----------
    complete_data: pd.DataFrame
        The complete data.
    gene_expression_data: pd.DataFrame
        The gene expression data.
    target_name: str (optional)
        The name of the target data.

    Returns:
    --------
    pd.DataFrame
        The flux data.
    """
    # Drop the target data from the full data
    complete_data = complete_data.drop(columns=[target_name])
    complete_data = complete_data.drop(columns="Row")
    # Drop the gene expression data from the full data
    result = complete_data.drop(columns=gene_expression_data.columns.values)

    return result


def get_target_training_data(
    complete_data: pd.DataFrame,
    testing_data_indices: list[int],
    target_name: str = TARGET_NAME,
) -> np.ndarray:
    """Get the target training data.

    Parameters:
    -----------
    complete_data: pd.DataFrame
        The complete data.
    testing_data_indices: list[int]
        The indices of the testing data.
    target_name: str (optional)
        The name of the target data.

    Returns:
    --------
    pd.Series
        The target training data.
    """
    target_data = complete_data[target_name]
    result = target_data.drop(target_data.index[testing_data_indices])
    result = result.astype(np.float32).to_numpy()

    return result


def get_target_testing_data(
    complete_data: pd.DataFrame,
    testing_data_indices: list[int],
    target_name: str = TARGET_NAME,
) -> np.ndarray:
    """Get the target testing data.

    Parameters:
    -----------
    complete_data: pd.DataFrame
        The complete data.
    testing_data_indices: list[int]
        The indices of the testing data.
    target_name: str (optional)
        The name of the target data.

    Returns:
    --------
    pd.Series
        The target testing data.
    """
    target_data = complete_data[target_name]
    result = target_data.iloc[testing_data_indices]
    result = result.astype(np.float32).to_numpy()

    return result


def get_complete_data(file_path: str) -> pd.DataFrame:
    """Get the complete data.

    Parameters:
    -----------
    file_path: str
        The path to the file containing the complete data.

    Returns:
    --------
    pd.DataFrame
    """
    complete_data: pd.DataFrame = pyreadr.read_r(file_path)[None]
    if complete_data is None:
        print("error with full data, not able to read")
    else:
        print("Success in loading complete data")
        print("Shape of the complete data:", complete_data.shape)
    # Remove columns consisting of only zeros
    result = complete_data.loc[:, (complete_data != 0).any(axis=0)]

    return result


def get_testing_data_indices(file_path: str) -> list[int]:
    """Get the indices of the testing data from the file at file_path.

    Parameters:
    -----------
    file_path: str
        The path to the file containing the indices of the testing data.

    Returns:
    --------
    list[int]
        The indices of the testing data.

    """
    with open(file_path, "r") as csvfile:
        result = []
        for row in csv.reader(csvfile, delimiter=";"):
            result.append(row[0])  # careful here with [0]

    result = list(map(int, result))  # fixed testing indexes

    return result


def scale_gene_expression_data(gene_expression_data: pd.DataFrame) -> np.ndarray:
    """Scale the gene expression data.

    Parameters:
    -----------
    gene_expression_data: pd.DataFrame
        The gene expression data.

    Returns:
    --------
    np.ndarray
        The scaled gene expression data.
    """
    gene_expression_scaler = preprocessing.StandardScaler().fit(gene_expression_data)
    result = gene_expression_scaler.transform(gene_expression_data).astype(np.float32)

    return result


def scale_flux_data(flux_data: pd.DataFrame) -> np.ndarray:
    """Scale the flux data.

    Parameters:
    -----------
    flux_data: pd.DataFrame
        The flux data.

    Returns:
    --------
    np.ndarray
        The scaled flux data.
    """
    flux_scaler = preprocessing.StandardScaler().fit(flux_data)
    result = flux_scaler.transform(flux_data).astype(np.float32)

    return result


def get_preprocessed_data(
    testing_data_indices_file_path: str,
    complete_data_file_path: str,
    gene_expression_data_file_path: str,
) -> MomaDataclass:
    """Get the data for scaled gene expression, scaled fluxes and target data (training & testing).

    Parameters:
    -----------
    testing_data_indices_file_path: str
        The path to the file containing the indices of the testing data.
    complete_data_file_path: str
        The path to the file containing the complete data.
    gene_expression_data_file_path: str
        The path to the file containing the gene expression data.

    Returns:
    --------
    dict[str, Union[np.ndarray, pd.Series]]
        A dictionary containing the scaled gene expression training data, scaled gene expression testing data, scaled flux training data, scaled flux testing data, target training data and target testing data.
    """
    testing_data_indices = get_testing_data_indices(
        file_path=testing_data_indices_file_path
    )

    complete_data = get_complete_data(file_path=complete_data_file_path)

    gene_expression_data = get_gene_expression_data(
        file_path=gene_expression_data_file_path
    )

    flux_data = get_flux_data(
        complete_data=complete_data, gene_expression_data=gene_expression_data
    )

    gene_expression_training_data = get_gene_expression_training_data(
        gene_expression_data=gene_expression_data,
        testing_data_indices=testing_data_indices,
    )
    gene_expression_testing_data = get_gene_expression_testing_data(
        gene_expression_data=gene_expression_data,
        testing_data_indices=testing_data_indices,
    )

    flux_training_data = get_flux_training_data(
        flux_data=flux_data, testing_data_indices=testing_data_indices
    )
    flux_testing_data = get_flux_testing_data(
        flux_data=flux_data, testing_data_indices=testing_data_indices
    )

    scaled_gene_expression_training_data = scale_gene_expression_data(
        gene_expression_data=gene_expression_training_data
    )
    scaled_gene_expression_testing_data = scale_gene_expression_data(
        gene_expression_data=gene_expression_testing_data
    )

    scaled_flux_training_data = scale_flux_data(flux_data=flux_training_data)
    scaled_flux_testing_data = scale_flux_data(flux_data=flux_testing_data)

    target_training_data = get_target_training_data(
        complete_data=complete_data, testing_data_indices=testing_data_indices
    )
    target_testing_data = get_target_testing_data(
        complete_data=complete_data, testing_data_indices=testing_data_indices
    )

    result = MomaDataclass(
        scaled_gene_expression_training_data=scaled_gene_expression_training_data,
        scaled_gene_expression_testing_data=scaled_gene_expression_testing_data,
        scaled_flux_training_data=scaled_flux_training_data,
        scaled_flux_testing_data=scaled_flux_testing_data,
        target_training_data=target_training_data,
        target_testing_data=target_testing_data,
    )
    return result
