import cobra
import json
import multiprocessing
import os
import pathlib
import pandas as pd


from typing import Union
from etfl.io import dict as etfl_dict
# from etfl.io import dict as etfl_dict
from bench.models import strategy


class Cefl(strategy.Strategy):
    """
    params: None
    
    cEFL model:
    - expression constraints
    - constant biomass composition
    """

    def __init__(self) -> None:
        model_path: str = "data/models/etfl/cEFL_20240417_1405.json"
        self.cefl_model: cobra.Model = self.load_json_model(filepath=pathlib.Path(model_path))

    def _predict_task1(self, index_row_tuple) -> tuple[int, Union[float, None]]:
        """
        params:
        - index: int: index of the row
        - row: pd.Series: row of the DataFrame
        return:
        - index: int: index of the row
        - solution.objective_value: float: predicted growth rate
        """
        index, row = index_row_tuple
        try:
            gene_id = row["knockout_gene_id"]
            gene = self.cefl_model.get_by_id(gene_id)
            gene.knock_out()
            solution = self.cefl_model.optimize()
            return index, solution.objective_value
        except:
            return index, None

    def predict_task1(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        params:
        - data: pd.DataFrame: gene knockout data
        return:
        - result: pd.DataFrame: predicted growth rate data

        Predict the growth rate given the gene knockout data
        """
        if os.path.exists("data/predictions/cefl/task1_results.csv"):
            print("\tUsing cached results")
            return pd.read_csv("data/predictions/cefl/task1_results.csv")

        print(f"\tUsing {multiprocessing.cpu_count()} cores")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(
                pool.imap(
                    self._predict_task1,
                    data.iterrows(),
                )
            )

        for index, result in results:
            data.at[index, "prediction"] = result

        print(f"Could not build model for {data['prediction'].isna().sum()} genes")
        data.to_csv("data/predictions/cefl/task1_results.csv", index=False)
        return data

    def predict_task2(self, data: list):
        raise NotImplementedError("This model does not support Task 2")

    def load_json_model(self, filepath: pathlib.Path) -> cobra.Model:
        """
        params:
        - filepath: pathlib.Path: path to the JSON file
        return:
        - model: cobra.Model: the cEFL model loaded from the JSON file
        """
        with open(filepath, "r") as file:
            json_object = json.load(file)

        model = etfl_dict.model_from_dict(obj=json_object, solver="optlang-gurobi")

        return model
