import numpy as np
import pandas as pd

from bench.models.moma import preprocessing


def get_ralser_data(cols_growth_data: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Get the Ralser proteomics and growth data.

    Parameters
    ----------
    cols_growth_data : dict[str, str]
        The columns of growth rates to extract

    Returns
    -------
    dict[str, pd.DataFrame]
        The proteomics and growth data.
        keys: "proteomics" and "growth"
    """
    proteomics_data_ralser = pd.read_csv("data/models/moma/yeast5k_impute_wide.csv")
    growth_rates_ralser = pd.read_csv("data/models/moma/yeast5k_growthrates_byORF.csv")

    # get only the growth rates of interest from cols_growth_data, and orf
    growth_rates_ralser = growth_rates_ralser[["orf", *cols_growth_data.keys()]]

    preprocessed_data = ralser_preprocessing(
        proteomics_data=proteomics_data_ralser,
        growth_data=growth_rates_ralser,
        col_to_rename=cols_growth_data,
    )

    return preprocessed_data


def ralser_preprocessing(
    proteomics_data: pd.DataFrame,
    growth_data: pd.DataFrame,
    col_to_rename: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Preprocess the Ralser dataset for the MOMA model.

    Parameters
    ----------
    proteomics_data : pd.DataFrame
        The proteomics data.
    growth_data : pd.DataFrame
        The growth data.

    Returns
    -------
    dict[str, pd.DataFrame]
        The preprocessed proteomics and growth data.
        keys: proteomics, growth
    """
    # PROTEOMICS
    proteomics_data = _remove_quality_controls(data=proteomics_data)
    proteomics_data = _remove_his3delta_controls(data=proteomics_data)
    # Rename the columns to the systematic gene names
    proteomics_data = _rename_ralser_columns(data=proteomics_data)
    # Set the first column as the index
    proteomics_data.set_index(proteomics_data.columns[0], inplace=True)
    proteomics_data.index.name = "protein_id"
    # Transpose the data so that the rows are knockouts and the columns are proteins
    proteomics_data = proteomics_data.T

    # GROWTH
    growth_data.rename(columns={"orf": "systematic_name"}, inplace=True)
    for key, value in col_to_rename.items():
        growth_data.rename(columns={key: value}, inplace=True)
    growth_data.set_index("systematic_name", inplace=True)

    # INTERSECTION - intersect two dataframes
    datasets = {"proteomics": proteomics_data, "growth": growth_data}
    preprocessed_data = preprocessing.filter_data(datasets=datasets)

    aligned_proteomics_data = preprocessed_data["proteomics"]
    aligned_growth_data = preprocessed_data["growth"]

    # remove duplicates - keep only one of the duplicates
    aligned_proteomics_data_no_duplicates = aligned_proteomics_data[
        ~aligned_proteomics_data.index.duplicated(keep="first")
    ]

    # sort the dataframes by the knockout name
    aligned_proteomics_data_no_duplicates = (
        aligned_proteomics_data_no_duplicates.sort_index()
    )
    aligned_growth_data = aligned_growth_data.sort_index()

    np.random.seed(42)
    shuffled_indices = np.random.permutation(
        aligned_proteomics_data_no_duplicates.index
    )
    shuffled_proteomics = aligned_proteomics_data_no_duplicates.loc[shuffled_indices]
    shuffled_growth = aligned_growth_data.loc[shuffled_indices]

    result = {
        "proteomics": shuffled_proteomics,
        "growth": shuffled_growth,
    }
    return result


def _remove_quality_controls(data: pd.DataFrame) -> pd.DataFrame:
    """Remove quality control columns from the dataset (_qc_qc_qc).

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to remove quality control columns from.

    Returns
    -------
    pd.DataFrame
        The dataset without quality control columns.
    """
    # Remove the quality control columns (they contain "_qc_qc_qc")
    quality_control_data = data.filter(regex="_qc_qc_qc", axis=1)
    result = data.drop(quality_control_data.columns, axis=1)
    return result


def _remove_his3delta_controls(data: pd.DataFrame) -> pd.DataFrame:
    """Remove the his3Δ control strains from the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to remove the his3Δ control strains from.

    Returns
    -------
    pd.DataFrame
        The dataset without the his3Δ control strains.
    """
    his3_control_data = data.filter(regex="_HIS3", axis=1)
    result = data.drop(his3_control_data.columns, axis=1)
    return result


def _rename_ralser_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns in the Ralser dataset to the systematic gene names.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to rename the columns for.

    Returns
    -------
    pd.DataFrame
        The dataset with renamed columns.
    """
    new_columns = [
        data.columns[i] if i < 1 else _rename_ralser_single_column(col=data.columns[i])
        for i in range(len(data.columns))
    ]
    data.columns = new_columns
    return data


def _rename_ralser_single_column(col: str) -> str:
    """Rename the original column in Ralser dataset to the systematic gene name.

    Parameters
    ----------
    col : str
        The original column name.

    Returns
    -------
    str
        The systematic gene name.
    """
    if "_ko_" in col:
        parts = col.split("_ko_")
        # Further split by "_" and take the first element which is the systematic gene name
        gene_name = parts[1].split("_")[0]
        return gene_name
    return col
