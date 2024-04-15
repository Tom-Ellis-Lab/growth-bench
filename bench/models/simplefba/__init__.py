"""
Model: Simple Baseline FBA model
https://www.ebi.ac.uk/biomodels/BIOMD0000001063#Overview
"""

from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from cobra.io import read_sbml_model

from bench.models.strategy import Strategy


class SimpleFBA(Strategy):
    model_path = "data/models/simplefba/yeast-GEM.xml"

    @staticmethod
    def _predict(args):
        index, row = args
        try:
            ko_model = read_sbml_model(SimpleFBA.model_path)
            gene_id = row["knockout_gene_id"]
            gene = ko_model.genes.get_by_id(gene_id)
            gene.knock_out()
            ko_solution = ko_model.optimize()
            return index, ko_solution.objective_value
        except:
            return index, None

    def predict_task1(self, data: pd.DataFrame) -> pd.DataFrame:
        import os

        # Check whether predictions have already been computed
        if os.path.exists("data/predictions/simplefba/task1_results.csv"):
            print("\tUsing cached results")
            return pd.read_csv("data/predictions/simplefba/task1_results.csv")

        print(f"\tUsing {mp.cpu_count()} cores")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(pool.imap(SimpleFBA._predict, data.iterrows()))

        # Update DataFrame with results
        for index, result in results:
            data.at[index, "prediction"] = result

        print(f"Could not build model for {data['prediction'].isna().sum()} genes")
        data.to_csv("data/predictions/simplefba/task1_results.csv", index=False)
        return data

    def predict_task2(self, data: List):
        raise NotImplementedError("This model does not support Task 2")
