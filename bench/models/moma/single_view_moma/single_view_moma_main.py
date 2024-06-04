import random
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing as sklearn_preprocessing
from sklearn.decomposition import PCA
import tensorflow as tf
import wandb

sys.path.append(".")

from bench.models.moma import model, train, preprocessing
from bench.models.moma.culley_moma import culley_main
from bench.models.moma.ralser_moma import ralser_preprocessing

config = {
    "epochs": 1000,
    "neurons": 1000,
    "batch_size": 256,
    "learning_rate": 0.001,
    "momentum": 0.75,
    "optimizer": "adagrad",  # "sgd", "adam", "adagrad"
    "n_outputs": 1,  # 1, 3
    "input_type": "proteomics",  # "proteomics", "transcriptomics", "fluxomics"
    "growth_rate_source": "ralser",  # "culley", "ralser"
    "pca_explained_variance": None,  # 0.9999,
    "save_weights": False,
}


def one_view_moma_main(config: wandb.Config) -> None:
    """Train a one-view model using the MOMA dataset.

    The model can be trained using the fluxomics, transcriptomics or proteomics data.

    Parameters
    ----------
    config : wandb.Config
        The configuration settings for the model.
    n_outputs : int, optional
        The number of outputs to predict, by default 1.
    growth_rate_source : str, optional
        The source of the growth rates, by default "ralser".
    """
    upper_case_name = config.input_type.upper()

    print(
        f"\n==== {upper_case_name} 1-VIEW {config.growth_rate_source.upper()}-INPUT {config.n_outputs}-OUTPUT MODEL ====\n"
    )
    print(
        f"\n==== USING {config.optimizer.upper()} OPTIMISER, LEARNING RATE {config.learning_rate} ====\n"
    )

    tf.random.set_seed(42)
    random.seed(42)

    growth_rate_options = {
        1: {"SC": "growth_rate"},
        3: {
            "SC": "growth_rate_SC",
            "SM": "growth_rate_SM",
            "YPD": "growth_rate_YPD",
        },
    }
    growth_rate_cols = growth_rate_options[config.n_outputs]

    if config.input_type in ["transcriptomics", "fluxomics"]:
        culley_data = culley_main.get_culley_data()
        input_data = culley_data[config.input_type]
        growth_data = culley_data["growth"]
    elif config.input_type == "proteomics":
        ralser_data = ralser_preprocessing.get_ralser_data(
            cols_growth_data=growth_rate_cols
        )
        input_data = ralser_data[config.input_type]
        growth_data = ralser_data["growth"]

    if (
        config.growth_rate_source == "ralser" and config.input_type != "proteomics"
    ) or (
        config.growth_rate_source == "culley"
        and config.input_type not in ["transcriptomics", "fluxomics"]
    ):

        data = preprocessing.intersect_input_data_with_growth_rates(
            input_data=input_data, growth_rate_data=growth_data
        )
        input_data = data["input_data"]
        growth_data = data["growth"]

    input_data = get_train_test_data(input_data=input_data, growth_data=growth_data)
    X_train = input_data["X_train"]
    y_train = input_data["y_train"]
    X_test = input_data["X_test"]
    y_test = input_data["y_test"]

    scaler = sklearn_preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()

    if config.pca_explained_variance is not None:
        print(
            f"\n==== RUNNING PCA FOR THRESHOLD: {config.pca_explained_variance} ====\n"
        )

        # PCA
        pca = PCA().fit(X_train)
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        X_train_pca = X_train_scaled.copy()
        X_test_pca = X_test_scaled.copy()
        n_components = (
            np.argmax(explained_variance_ratio >= config.pca_explained_variance) + 1
        )
        print(
            f"Threshold: {config.pca_explained_variance}, number of components to keep: {n_components}"
        )
        pca_results = preprocessing.apply_pca_on_data(
            X_train=X_train_pca, X_test=X_test_pca, n_components=n_components
        )
        x_train_pca = pca_results["train"]
        x_test_pca = pca_results["test"]

        print(
            f"Threshold: {config.pca_explained_variance}: shape of X_train {X_train.shape} and X_train_pca {x_train_pca.shape}"
        )
        print(
            f"Threshold: {config.pca_explained_variance}: shape of X_test {X_test.shape} and X_test_pca {x_test_pca.shape}"
        )
        X_train = x_train_pca
        X_test = x_test_pca

    single_view_model = model.init_single_view_model(
        input_dim=X_train_scaled.shape[1],
        model_name=config.input_type,
        input_neurons=config.neurons,
        output_neurons=config.n_outputs,
    )

    optimiser = preprocessing.get_optimiser(config=config)

    single_view_model.compile(
        loss="mean_squared_error", optimizer=optimiser, metrics=["mean_absolute_error"]
    )

    history = single_view_model.fit(
        x=X_train_scaled,
        y=y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_test_scaled, y_test),
        verbose=0,
    )

    if config.save_weights:
        print("\n==== SAVING WEIGHTS ====\n")
        single_view_model.save_weights(
            f"data/models/moma/{config.input_type}_{config.n_outputs}_output.weights.h5"
        )

    total_samples = len(X_train)  # Total number of samples in the training set
    normalized_loss = [
        loss * config.batch_size / total_samples for loss in history.history["loss"]
    ]
    normalized_val_loss = [
        val_loss * config.batch_size / len(X_test)
        for val_loss in history.history["val_loss"]
    ]

    train.plot_loss(
        loss=normalized_loss,
        val_loss=normalized_val_loss,
        plot_to_save_dir="data/models/moma/",
        name=f"{config.input_type}_model_loss",
    )
    if config.n_outputs == 1:
        results = train.evaluate(
            model=single_view_model,
            X_test=X_test_scaled,
            y_test=y_test,
        )
    else:
        results = train.evaluate(
            model=single_view_model,
            X_test=X_test_scaled,
            y_test=y_test,
            n_outputs=config.n_outputs,
        )

    print(
        f"\n==== {upper_case_name} 1-VIEW MODEL {config.growth_rate_source.upper()}-INPUT {config.n_outputs}-OUTPUT RESULTS ====\n"
    )
    print(f"\n==== OPTIMIZER USED: {config.optimizer} ====\n")
    for key, value in results.items():
        print(f"{key}: {value}")

    wandb.log(results)


def get_train_test_data(
    input_data: pd.DataFrame, growth_data: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Split the input data into training and testing sets.

    Parameters
    ----------
    input_data : pd.DataFrame
        The input data to split.
    growth_data : pd.DataFrame
        The growth data to split.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and testing data.
        keys: X_train, y_train, X_test, y_test, train_indices, test_indices
    """

    random_state = 42
    test_size = 0.2

    # Generate the split indices using one of the datasets
    split_result = train.random_split(
        input_data,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices = split_result["train"]
    test_indices = split_result["test"]

    input_data = train.apply_indices_split(
        data=input_data,
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )

    growth_data = train.apply_indices_split(
        data=growth_data,
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )

    X_train = input_data["train"]
    y_train = growth_data["train"]
    X_test = input_data["test"]
    y_test = growth_data["test"]

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
    return result


if __name__ == "__main__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="growth-bench",
        # track hyperparameters and run metadata with wandb.config
        config=config,
    )
    one_view_moma_main(config=wandb.config)
    print("\n==== DONE ====\n")
