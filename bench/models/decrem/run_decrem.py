import multiprocessing
import os
import sys
from typing import Union
from functools import partial

import cobra
import scipy
import tqdm
import numpy as np
import pandas as pd

sys.path.append("../../../")

from temp_tools import matlab_model_gateway
from bench.models.decrem import decrem_gateway


# cobra model, print number of reactions, metabolites, genes, variables, constraints
# print(model)
# print("Reactions", len(model.reactions))
# print("Metabolites", len(model.metabolites))
# print("Genes", len(model.genes))
# print("Variables", len(model.variables))
# print("Constraints", len(model.constraints))


# print(model.objective)
# reaction = model.reactions.get_by_id("BIOMASS_SC5_notrace")
# print(reaction.name)
def _predict_task1(model, index_row_tuple) -> tuple[int, Union[float, None]]:
    """
    Parameters
    ----------
    index_row_tuple: tuple[int, pd.Series]
        index of the row and the row of the DataFrame

    Returns
    -------
    tuple[int, Union[float, None]]
        index of the row and the predicted growth rate
    """
    index, row = index_row_tuple
    try:
        gene_id = row["knockout_gene_id"]
        print(gene_id)
        gene = model.genes.get_by_id(gene_id)
        gene.knock_out()
        solution = model.optimize()
        return index, solution.objective_value
    except:
        return index, None


model_name = "decrem"


def predict_task1(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    data: pd.DataFrame
        gene knockout data

    Returns
    -------
    pd.DataFrame
        predicted growth rate data

    Predict the growth rate given the gene knockout data
    """
    results_path = "../../../data/predictions/" + model_name + "/task1_results.csv"
    if os.path.exists(results_path):
        print("\tUsing cached results")
        return pd.read_csv(results_path)
    print(f"\tUsing {multiprocessing.cpu_count()} cores")
    partial_predict_task1 = partial(_predict_task1, model=model)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(partial_predict_task1, data.iterrows()), total=len(data)
            )
        )

    for index, result in results:
        data.at[index, "prediction"] = result

    print(f"Could not build model for {data['prediction'].isna().sum()} genes")
    data.to_csv(results_path, index=False)
    return data


if __name__ == "__main__":
    # Load and process the data
    # MODEL_PATH = "data/models/decrem/Yeast_linear_model.mat"
    MODEL_PATH = "../../../data/models/decrem/Yeast_linear_model.mat"
    model = decrem_gateway.load_matlab_model(MODEL_PATH)
    data = pd.read_csv("../../../data/tasks/task1/growth_rate.csv")

    # rename column to true
    data.rename(
        columns={
            "hap a | growth (exponential growth rate) | standard | minimal complete | Warringer J~Blomberg A, 2003": "true",
            "orf": "knockout_gene_id",
        },
        inplace=True,
    )

    # Predict on dataset
    result = predict_task1(model, data)

    # Calculate performance metric (here, MSE and pearson correlation)
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr

    results_notna = result.dropna()

    mse = mean_squared_error(results_notna["true"], results_notna["prediction"])
    pearson = pearsonr(results_notna["true"], results_notna["prediction"])[0]
    spearman = results_notna["true"].corr(
        results_notna["prediction"], method="spearman"
    )

    result = {
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        "coverage": 1 - result["prediction"].isna().sum() / len(result),
    }
    print(result)
