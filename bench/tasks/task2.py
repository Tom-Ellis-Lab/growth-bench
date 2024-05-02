import scipy
import numpy as np
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
    def strategy_params(self) -> strategy_.SetupParams:
        """Get the strategy parameters

        Returns
        -------
        strategy.SetupParams
            A dataclass object containing the strategy parameters
        """
        if isinstance(self.strategy, moma_strategy.MomaStrategy):
            self._strategy_params = self.moma_params
        elif isinstance(self.strategy, lasso_strategy.LassoStrategy):
            self._strategy_params = self.lasso_params
        else:
            self._strategy_params = strategy_.SetupParams(
                gene_expression_data=None,
                flux_data=None,
            )
        return self._strategy_params

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
        """Get LASSO parameters

        Returns
        -------
        lasso_strategy.LassoParams
            A dataclass object containing the LASSO parameters
        """

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

        prediction = self.strategy.predict_task2(
            data=[
                self.strategy_params.flux_data,
                self.strategy_params.gene_expression_data,
            ]
        )

        Y_test = self.strategy_params.target_data

        if prediction.ndim > 1:
            prediction = prediction.ravel()  # Flatten the array

        # Check for NaN and handle them
        Y_test_ = np.nan_to_num(
            Y_test
        )  # Convert NaN to zero (or another specified value)
        prediction_ = np.nan_to_num(prediction)

        mse = float(metrics.mean_squared_error(y_true=Y_test_, y_pred=prediction_))
        r_squared = self.r_squared(Y=Y_test_, predictions=prediction_)
        pearson, _ = scipy.stats.pearsonr(Y_test_, prediction_)
        spearman, _ = scipy.stats.spearmanr(Y_test_, prediction_)

        # Calculate coverage
        non_nan_count = np.count_nonzero(~np.isnan(prediction_))
        total_count = len(prediction_)
        coverage = float(non_nan_count / total_count)

        result = {
            "mse": mse,
            "pearson": pearson,
            "spearman": spearman,
            "coverage": coverage,
            "r_squared": r_squared,
        }

        return result

    @staticmethod
    def r_squared(Y: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate the coefficient of determination R^2.

        Parameters
        ----------
        Y : np.ndarray
            The actual target data.
        predictions : np.ndarray
            The predicted data from the model.

        Returns
        -------
        float
            The calculated R^2 value.
        """
        residual = Y - predictions
        residual_sum_of_squares = np.sum(np.square(residual))
        total_sum_of_squares = np.sum(np.square(Y - np.mean(Y)))

        return 1 - (residual_sum_of_squares / total_sum_of_squares)
