import numpy as np
import pandas as pd


def culley_preprocessing(
    transcriptomics_data: pd.DataFrame,
    fluxomics_data: pd.DataFrame,
    growth_data: pd.DataFrame,
    mapping_dict: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Preprocess the Culley dataset for the MOMA model.

    Parameters
    ----------
    transcriptomics_data : pd.DataFrame
        The transcriptomics data.
    fluxomics_data : pd.DataFrame
        The fluxomics data.
    growth_data : pd.DataFrame
        The growth data.
    mapping_dict : dict[str, str]
        The mapping between standard and systematic names.

    Returns
    -------
    dict[str, pd.DataFrame]
        The preprocessed transcriptomics, fluxomics, and growth data.
        keys: transcriptomics, fluxomics, growth
    """

    # Transciptomics
    transcriptomics_data["knockout_id"] = (
        transcriptomics_data["knockout_id"]
        .map(mapping_dict)
        .fillna(transcriptomics_data["knockout_id"])
    )
    transcriptomics_data = transcriptomics_data.loc[
        :, (transcriptomics_data != 0).any(axis=0)
    ]
    transcriptomics_data.drop(columns=["log2relT"], inplace=True)
    transcriptomics_data.set_index("knockout_id", inplace=True)

    # Fluxomics
    fluxomics_data = fluxomics_data.loc[:, (fluxomics_data != 0).any(axis=0)]
    fluxomics_data.rename(columns={"Row": "knockout_id"}, inplace=True)
    # Convert to sytematic names and get names and growth rates
    fluxomics_data["knockout_id"] = (
        fluxomics_data["knockout_id"]
        .map(mapping_dict)
        .fillna(fluxomics_data["knockout_id"])
    )
    fluxomics_data.set_index("knockout_id", inplace=True)

    # Growth
    growth_data.rename(
        columns={"Row": "knockout_id", "log2relT": "growth_rate"}, inplace=True
    )
    growth_data["knockout_id"] = (
        growth_data["knockout_id"].map(mapping_dict).fillna(growth_data["knockout_id"])
    )
    growth_data.set_index("knockout_id", inplace=True)

    # Finding the Intersection of Knockouts
    common_knockouts = set(transcriptomics_data.index).intersection(
        fluxomics_data.index, growth_data.index
    )

    # Filtering Dataframes by the Intersection
    transcriptomics_data = transcriptomics_data[
        transcriptomics_data.index.isin(common_knockouts)
    ]
    fluxomics_data = fluxomics_data[fluxomics_data.index.isin(common_knockouts)]
    growth_data = growth_data[growth_data.index.isin(common_knockouts)]

    # Sorting Dataframes by Knockout ID
    transcriptomics_data = transcriptomics_data.sort_index()
    fluxomics_data = fluxomics_data.sort_index()
    growth_data = growth_data.sort_index()

    # Shuffling the Data
    shuffled_indices = np.random.permutation(transcriptomics_data.index)
    shuffled_transcriptomics = transcriptomics_data.loc[shuffled_indices]
    shuffled_fluxomics = fluxomics_data.loc[shuffled_indices]
    shuffled_growth = growth_data.loc[shuffled_indices]

    result = {
        "transcriptomics": shuffled_transcriptomics,
        "fluxomics": shuffled_fluxomics,
        "growth": shuffled_growth,
    }
    return result
