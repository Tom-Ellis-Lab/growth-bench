import time

import cvxpy
import numpy as np

import gurobipy # necessary to run the optimisation


def loss(
    X: np.ndarray,
    Y: np.ndarray,
    beta_gene_expression: cvxpy.Variable,
    beta_fluxes: cvxpy.Variable,
    intercept_value: cvxpy.Parameter,
) -> cvxpy.Expression:
    """Loss function for the regression problem.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    Y : np.ndarray
        The target data.
    beta_gene_expression : cvxpy.Variable
        The beta vector for the gene expression data.
    beta_fluxes : cvxpy.Variable
        The beta vector for the flux data.
    intercept_value : cvxpy.Parameter
        The intercept value.

    Returns
    -------
    cvxpy.Expression
        The loss function.
    """
    beta = cvxpy.hstack(
        [beta_gene_expression, beta_fluxes]
    )  # Horizontal concatenation of an arbitrary number of Expressions.
    intercept = np.repeat(intercept_value.value, Y.shape[0])
    x_beta = cvxpy.matmul(X, beta)
    Y_minus_x_beta = Y - x_beta
    Y_minus_x_beta_minus_intercept = Y_minus_x_beta - intercept
    norm2 = cvxpy.norm2(Y_minus_x_beta_minus_intercept)
    return norm2**2


def modality_regularizer(beta: cvxpy.Variable) -> cvxpy.Expression:
    """L1 norm of the beta vector.

    Parameters
    ----------
    beta : cvxpy.Variable
        The beta vector.

    Returns
    -------
    cvxpy.Expression
        The L1 norm of the beta vector.
    """

    return cvxpy.norm1(beta)


def group_regularizer(
    groups: list[tuple[list[cvxpy.Variable], int]], *regularization_weights
):
    """Group Lasso regularizer.

    Parameters
    ----------
    groups : list[tuple[list[cvxpy.Variable], int]]
        A list of tuples where each tuple contains a list of cvxpy.Variable objects and an integer.
    regularization_weights : list[float]
        A list of weights for each group.

    Returns
    -------
    cvxpy.Expression
        The Group Lasso regularizer.
    """
    expression = []
    if len(regularization_weights) != len(
        groups
    ):  # if we do not want/cannot provide a lambda for each group
        remaining_groups = np.ones(len(groups) - len(regularization_weights))
        regularization_weights = list(regularization_weights)
        regularization_weights.extend(remaining_groups)

    for index, group in enumerate(groups):
        if not group[0]:  # Check if the group is empty
            continue  # Skip this iteration if the group is empty
        expression.append(
            regularization_weights[index] * cvxpy.norm2(cvxpy.hstack(group[0]))
        )  # we assume every arg is a lambda
    if not expression:  # Check if the expression list is empty
        return 0
    result = sum(expression)

    return result


def assign_variables_to_groups(
    beta_vector: cvxpy.Variable, indices: list[int]
) -> list[tuple[list[cvxpy.Variable], int]]:
    """Assigns variables to groups.

    Parameters
    ----------
    beta_vector : cvxpy.Variable
        The beta vector.
    indices : list[int]
        The indices of the groups.

    Returns
    -------
    list[tuple[list[cvxpy.Variable], int]]
        A list of tuples where each tuple contains a list of cvxpy.Variable objects and an integer.
    """
    number_of_groups = len(set(indices))
    groups = []
    for i in range(
        1, number_of_groups + 1
    ):  # we are assuming each group is given a number starting from 1
        idxs = (np.asarray(indices) == i).nonzero()
        groups.append(([beta_vector[idx] for idx in idxs[0]], i))
    return groups


def objective_function(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    beta_gene_expression: cvxpy.Variable,
    beta_fluxes: cvxpy.Variable,
    lambda_gene_expression: cvxpy.Parameter,
    lambda_fluxes: cvxpy.Parameter,
    intercepts: cvxpy.Parameter,
    indices_fluxes: list[int],
    indices_gene_expression: list[int],
):
    """Objective function for the regression problem.

    Parameters
    ----------
    X_train : np.ndarray
        The input data.
    Y_train : np.ndarray
        The target data.
    beta_gene_expression : cvxpy.Variable
        The beta vector for the gene expression data.
    beta_fluxes : cvxpy.Variable
        The beta vector for the flux data.
    lambda_gene_expression : cvxpy.Parameter
        The lambda value for the gene expression data.
    lambda_fluxes : cvxpy.Parameter
        The lambda value for the flux data.
    intercepts : cvxpy.Parameter
        The intercept value.
    indices_fluxes : list[int]
        The indices of the flux data.
    indices_gene_expression : list[int]
        The indices of the gene expression data.

    Returns
    -------
    cvxpy.Expression
        The objective function.
    """
    return (
        loss(
            X=X_train,
            Y=Y_train,
            beta_gene_expression=beta_gene_expression,
            beta_fluxes=beta_fluxes,
            intercept_value=intercepts,
        )
        + lambda_gene_expression * modality_regularizer(beta=beta_gene_expression)
        + lambda_fluxes * modality_regularizer(beta=beta_fluxes)
        + group_regularizer(
            assign_variables_to_groups(
                beta_vector=beta_gene_expression, indices=indices_fluxes
            ),
            1,
        )
        + group_regularizer(
            assign_variables_to_groups(
                beta_vector=beta_fluxes, indices=indices_gene_expression
            ),
            1,
        )
    )


def mse(
    X: np.ndarray,
    Y: np.ndarray,
    beta_ge: cvxpy.Variable,
    beta_mf: cvxpy.Variable,
    intercept_value: cvxpy.Parameter,
) -> float:
    """Mean squared error.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    Y : np.ndarray
        The target data.
    beta_ge : cvxpy.Variable
        The beta vector for the gene expression data.
    beta_mf : cvxpy.Variable
        The beta vector for the flux data.
    intercept_value : cvxpy.Parameter
        The intercept value.

    Returns
    -------
    float
        The mean squared error.
    """
    return (1.0 / X.shape[0]) * loss(X, Y, beta_ge, beta_mf, intercept_value).value


def optimise(
    process: int,
    lambdas_gene_expression: float,
    lambdas_fluxes: float,
    intercept: float,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    flux_dimensions_index: int,
    gene_dimensions_index: int,
    indices_fluxes: list[int],
    indices_gene_expression: list[int],
) -> dict[str, float]:
    """Create and solve the optimisation problem.

    Parameters
    ----------
    process : int
        The process number.
    lambdas_gene_expression : float
        The lambda value for the gene expression data.
    lambdas_fluxes : float
        The lambda value for the flux data.
    intercept : float
        The intercept value.
    X_train : np.ndarray
        The input data for training.
    Y_train : np.ndarray
        The target data for training.
    X_test : np.ndarray
        The input data for testing.
    Y_test : np.ndarray
        The target data for testing.
    flux_dimensions_index : int
        The index of the flux dimensions.
    gene_dimensions_index : int
        The index of the gene expression dimensions.
    indices_fluxes : list[int]
        The indices of the flux data.
    indices_gene_expression : list[int]
        The indices of the gene expression data.

    Returns
    -------
    dict[str, float]
        The results of the optimisation.
    """
    # Variables
    beta_gene_expression = cvxpy.Variable(flux_dimensions_index + 1)
    beta_fluxes = cvxpy.Variable(gene_dimensions_index + 1)

    # Parameters
    lambda_gene_expression_param = cvxpy.Parameter(nonneg=True)
    lambda_fluxes_param = cvxpy.Parameter(nonneg=True)
    intercept_param = cvxpy.Parameter()
    lambda_gene_expression_param.value = lambdas_gene_expression
    lambda_fluxes_param.value = lambdas_fluxes
    intercept_param.value = intercept

    print(
        "Process: {}, Lambda GE: {}, Lambda MF: {}, Intercept: {}".format(
            process,
            lambda_gene_expression_param.value,
            lambda_fluxes_param.value,
            intercept_param.value,
        )
    )
    substart_time = time.time()
    problem = cvxpy.Problem(
        cvxpy.Minimize(
            objective_function(
                X_train,
                Y_train,
                beta_gene_expression,
                beta_fluxes,
                lambda_gene_expression_param,
                lambda_fluxes_param,
                intercept_param,
                indices_fluxes,
                indices_gene_expression,
            )
        )
    )
    problem.solve(solver=cvxpy.GUROBI)

    train_error = mse(
        X=X_train,
        Y=Y_train,
        beta_ge=beta_gene_expression,
        beta_mf=beta_fluxes,
        intercept_value=intercept_param,
    )
    test_error = mse(
        X=X_test,
        Y=Y_test,
        beta_ge=beta_gene_expression,
        beta_mf=beta_fluxes,
        intercept_value=intercept_param,
    )
    total_time = (time.time() - substart_time) / 60
    print(
        "Process: {}, Total time in minutes: {}, Train error:{}, Test error: {}".format(
            process, total_time, train_error, test_error
        )
    )
    result = {
        "process": process,
        "lambda_ge": lambda_gene_expression_param.value,
        "lambda_mf": lambda_fluxes_param.value,
        "intercept": intercept_param.value,
        "train_error": train_error,
        "test_error": test_error,
        "total_time": total_time,
        "beta_ge": list(beta_gene_expression.value),
        "beta_mf": list(beta_fluxes.value),
    }
    return result
