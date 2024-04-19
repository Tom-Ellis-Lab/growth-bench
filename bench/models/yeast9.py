import cobra
import multiprocessing
import os
from tqdm import tqdm

import pandas as pd
from typing import Union

from bench.models import strategy
from temp_tools import matlab_model_gateway


class Yeast9(strategy.Strategy):
    """GEM taken from: https://github.com/SysBioChalmers/yeast-GEM/tree/main/model

    Parameters
    ----------
    None

    Attributes
    ----------
    model_path: str
        path to the yeast9 model
    yeast9_model: cobra.Model
        yeast9 model
    """

    def __init__(self) -> None:
        self.model_path: str = "data/models/yeast9/yeast-GEM.mat"
        self.yeast9_model: cobra.Model = matlab_model_gateway.load_model_from_mat(
            file_path=self.model_path
        )

    def _predict_task1(self, index_row_tuple) -> tuple[int, Union[float, None]]:
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
            gene = self.yeast9_model.genes.get_by_id(gene_id)
            gene.knock_out()
            solution = self.yeast9_model.optimize()
            return index, solution.objective_value
        except:
            return index, None

    def predict_task1(self, data: pd.DataFrame) -> pd.DataFrame:
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
        if os.path.exists("data/predictions/yeast9/task1_results.csv"):
            print("\tUsing cached results")
            return pd.read_csv("data/predictions/yeast9/task1_results.csv")

        print(f"\tUsing {multiprocessing.cpu_count()} cores")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(tqdm(pool.imap(self._predict_task1, data.iterrows()), total=len(data)))

        for index, result in results:
            data.at[index, "prediction"] = result

        print(f"Could not build model for {data['prediction'].isna().sum()} genes")
        data.to_csv("data/predictions/yeast9/task1_results.csv", index=False)
        return data

    def predict_task2(self, data: list):
        raise NotImplementedError("This model does not support Task 2")
