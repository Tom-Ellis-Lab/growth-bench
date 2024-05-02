import scipy
import numpy as np
import pandas as pd
from sklearn import metrics


from bench.tasks import task1
from bench.models import strategy as strategy_
from bench.models.moma import moma_strategy, moma_data_gateway
from bench.models.lasso import lasso_strategy


class Task2(task1.Task):
    """
    Task for running models on gene expression and flux data
    """

    _momma_data_dir = "data/models/moma/"
    _lasso_data_dir = "data/models/lasso/"
    _testing_data_indices_file_path = _momma_data_dir + "indices_for_testing_data.csv"
    _complete_data_file_path = _momma_data_dir + "complete_dataset.RDS"
    _gene_expression_data_file_path = _momma_data_dir + "gene_expression_dataset.RDS"

    def __init__(self, strategy: strategy_.MultiOmicsStrategy) -> None:
        """Initialize Task2 with a transcriptomics strategy

        Parameters
        ----------
        strategy : strategy.MultiOmicsStrategy
            A strategy object that implements the Strategy interface

        Returns
        -------
        None
        """
        self._gateway = moma_data_gateway
        self.strategy = strategy

    @property
    def moma_params(self) -> moma_strategy.MomaParams:
        """Get MOMA parameters

        Returns
        -------
        moma_strategy.MomaParams
            A dataclass object containing the MOMA parameters
        """
        if not hasattr(self, "_moma_params"):
            preprocessed_data: moma_data_gateway.MomaDataclass = (
                self._gateway.get_preprocessed_data(
                    testing_data_indices_file_path=self._testing_data_indices_file_path,
                    complete_data_file_path=self._complete_data_file_path,
                    gene_expression_data_file_path=self._gene_expression_data_file_path,
                )
            )
            scaled_gene_expression_testing_data = (
                preprocessed_data.scaled_gene_expression_testing_data
            )
            scaled_flux_testing_data = preprocessed_data.scaled_flux_testing_data

            self._moma_params = moma_strategy.MomaParams(
                multi_model_weights_path=self._momma_data_dir
                + "multi_view_model_GE_MF_0.weights.h5",
                gene_expression_data=scaled_gene_expression_testing_data,
                flux_data=scaled_flux_testing_data,
                gene_expression_weights_path=self._momma_data_dir
                + "gene_expression_weights.h5",
                flux_weights_path=self._momma_data_dir + "fluxomic_weights.h5",
                learning_rate=0.005,
                epochs=1000,
                momentum=0.75,
                neurons_single_view=1000,
                neurons_multi_view=15,
                target_data=preprocessed_data.target_testing_data,
            )

        return self._moma_params

    @property
    def lasso_params(self) -> lasso_strategy.LassoParams:

        if not hasattr(self, "_lasso_params"):
            preprocessed_data = self._gateway.get_preprocessed_data(
                testing_data_indices_file_path=self._testing_data_indices_file_path,
                complete_data_file_path=self._complete_data_file_path,
                gene_expression_data_file_path=self._gene_expression_data_file_path,
            )
            scaled_gene_expression_testing_data = (
                preprocessed_data.scaled_gene_expression_testing_data
            )
            scaled_flux_testing_data = preprocessed_data.scaled_flux_testing_data

            beta_gene_expression = np.genfromtxt(
                self._lasso_data_dir + "gene_expression_coefficients.csv"
            )

            beta_fluxes = np.genfromtxt(self._lasso_data_dir + "flux_coefficients.csv")

            result = lasso_strategy.LassoParams(
                gene_expression_data=scaled_gene_expression_testing_data,
                flux_data=scaled_flux_testing_data,
                target_data=preprocessed_data.target_testing_data,
                beta_gene_expression=beta_gene_expression,
                beta_fluxes=beta_fluxes,
                intercept=np.logspace(-1.6, 0.04, 12)[4],
            )
            self._lasso_params = result

        return self._lasso_params

    @property
    def strategy(self) -> strategy_.MultiOmicsStrategy:
        """Get the strategy object

        Returns
        -------
        strategy.MultiOmicsStrategy
            A strategy object that implements the Strategy interface
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: strategy_.MultiOmicsStrategy) -> None:
        """Set the strategy object

        Parameters
        ----------
        strategy : strategy.MultiOmicsStrategy
            A strategy object that implements the MultiOmicsStrategy interface

        Returns
        -------
        None
        """
        if isinstance(strategy, moma_strategy.MomaStrategy):
            self._strategy = strategy
            self._strategy.setup_strategy(params=self.moma_params)

        elif isinstance(strategy, lasso_strategy.LassoStrategy):
            self._strategy = strategy
            self._strategy.setup_strategy(params=self.lasso_params)
        else:
            self._strategy = strategy

    def benchmark(self) -> dict[str, float]:
        """Run the benchmark for Task2

        Returns
        -------
        dict[str, float]
            A dictionary containing the benchmark results (MSE, Pearson, Spearman, Coverage, R-squared)
        """

        moma_params = self.moma_params
        prediction = self.strategy.predict_task2(
            data=[moma_params.flux_data, moma_params.gene_expression_data]
        )

        Y_test = pd.Series(moma_params.target_data)
        if prediction.ndim > 1:
            prediction = prediction.ravel()  # Flatten the array
        prediction = pd.Series(prediction)

        mse = float(metrics.mean_squared_error(y_true=Y_test, y_pred=prediction))
        r_squared = float(metrics.r2_score(y_true=Y_test, y_pred=prediction))
        pearson = float(scipy.stats.pearsonr(x=Y_test, y=prediction)[0])
        spearman = float(Y_test.corr(prediction, method="spearman"))
        coverage = float(1 - prediction.isna().sum() / len(prediction + Y_test))

        result = {
            "mse": mse,
            "pearson": pearson,
            "spearman": spearman,
            "coverage": coverage,
            "r_squared": r_squared,
        }

        return result
