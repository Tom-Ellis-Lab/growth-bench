import random
import sys

import pandas as pd
from sklearn import preprocessing as sklearn_preprocessing
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

sys.path.append(".")

from bench.models.moma import model, train, preprocessing
from bench.models.moma.culley_moma import culley_main
from bench.models.moma.ralser_moma import ralser_preprocessing

config = {
    "epochs": 3000,
    "neurons": 1000,
    "batch_size": 256,
    "learning_rate": 0.015,
    "momentum": 0.75,
    "optimizer": "adagrad",
    "n_outputs": 1,
    "transcriptomics_weights_filename": "gene_expression_weights.h5",
    "fluxomics_weights_filename": "fluxomics_newly_trained.weights.h5",
    "proteomics_weights_filename": "proteomics_adagrad.weights.h5",
    "weights_filename": "three_view",
}


def three_view_moma(
    config: wandb.Config,
) -> None:
    """Train a three-view model using the MOMA dataset.
    The model is trained using the transcriptomics, fluxomics, and proteomics data.

    """
    tf.random.set_seed(42)
    random.seed(42)
    print(f"\n==== THREE-VIEW MOMA {config.n_outputs}-OUTPUT MODEL ====\n")

    if config.n_outputs == 1:
        cols_growth_data = {
            "SC": "growth_rate",
        }
    elif config.n_outputs == 3:
        cols_growth_data = {
            "SC": "growth_rate_SC",
            "SM": "growth_rate_SM",
            "YPD": "growth_rate_YPD",
        }
    culley_preprocessed_data: dict[str, pd.DataFrame] = culley_main.get_culley_data()
    ralser_preprocessed_data = ralser_preprocessing.get_ralser_data(
        cols_growth_data=cols_growth_data
    )
    data = {
        "transcriptomics": culley_preprocessed_data["transcriptomics"],
        "fluxomics": culley_preprocessed_data["fluxomics"],
        "growth": culley_preprocessed_data["growth"],
        "proteomics": ralser_preprocessed_data["proteomics"],
    }

    filtered_data = preprocessing.filter_data(
        datasets=data,
    )
    transcriptomics = filtered_data["transcriptomics"]
    fluxomics = filtered_data["fluxomics"]
    growth = filtered_data["growth"]
    proteomics = filtered_data["proteomics"]

    data = split_data(
        transcriptomics=transcriptomics,
        fluxomics=fluxomics,
        growth=growth,
        proteomics=proteomics,
        random_state=42,
        test_size=0.2,
    )
    transcriptomics_train = data["transcriptomics_train"]
    transcriptomics_test = data["transcriptomics_test"]
    fluxomics_train = data["fluxomics_train"]
    fluxomics_test = data["fluxomics_test"]
    proteomics_train = data["proteomics_train"]
    proteomics_test = data["proteomics_test"]
    growth_train = data["growth_train"]
    growth_test = data["growth_test"]

    X_train = pd.concat(
        [fluxomics_train, transcriptomics_train, proteomics_train], axis=1
    )
    X_test = pd.concat([fluxomics_test, transcriptomics_test, proteomics_test], axis=1)

    scaler = sklearn_preprocessing.StandardScaler().fit(transcriptomics_train)
    transcriptomics_train = scaler.transform(transcriptomics_train)
    transcriptomics_test = scaler.transform(transcriptomics_test)

    scaler = sklearn_preprocessing.StandardScaler().fit(fluxomics_train)
    fluxomics_train = scaler.transform(fluxomics_train)
    fluxomics_test = scaler.transform(fluxomics_test)

    scaler = sklearn_preprocessing.StandardScaler().fit(proteomics_train)
    proteomics_train = scaler.transform(proteomics_train)
    proteomics_test = scaler.transform(proteomics_test)

    y_test = growth_test.to_numpy()
    y_train = growth_train.to_numpy()

    transcriptomics_model = model.init_single_view_model(
        input_dim=transcriptomics_train.shape[1],
        model_name="transcriptomics",
        input_neurons=config.neurons,
        output_neurons=config.n_outputs,
    )

    transcriptomics_model.load_weights(
        f"data/models/moma/{config.transcriptomics_weights_filename}"
    )

    fluxomics_model = model.init_single_view_model(
        input_dim=fluxomics_train.shape[1],
        model_name="fluxomics",
        input_neurons=config.neuro  ns,
        output_neurons=config.n_outputs,
    )

    fluxomics_model.load_weights(
        f"data/models/moma/{config.fluxomics_weights_filename}"
    )

    proteomics_model = model.init_single_view_model(
        input_dim=proteomics_train.shape[1],
        model_name="proteomics",
        input_neurons=config.neurons,
        output_neurons=config.n_outputs,
    )

    proteomics_model.load_weights(
        f"data/models/moma/{config.proteomics_weights_filename}"
    )

    triple_view_model = model.init_triple_view_model(
        input1_dim=fluxomics_train.shape[1],
        input2_dim=transcriptomics_train.shape[1],
        input3_dim=proteomics_train.shape[1],
        neurons=config.neurons,
        model_1=fluxomics_model,
        model_2=transcriptomics_model,
        model_3=proteomics_model,
    )

    optimiser = preprocessing.get_optimiser(config=config)

    triple_view_model.compile(
        loss="mean_squared_error",
        optimizer=optimiser,
        metrics=["mean_absolute_error"],
    )
    history = triple_view_model.fit(
        x=[fluxomics_train, transcriptomics_train, proteomics_train],
        y=y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(
            [fluxomics_test, transcriptomics_test, proteomics_test],
            y_test,
        ),
        verbose="auto",
        callbacks=[
            WandbMetricsLogger(),
        ],
    )
    if config.weights_filename is not "":
        triple_view_model.save_weights(
            f"data/models/moma/{config.weights_filename}_{config.n_outputs}_model.weights.h5"
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
        name="three_view_model_loss",
    )
    results = train.evaluate(
        model=triple_view_model,
        X_test=[fluxomics_test, transcriptomics_test, proteomics_test],
        y_test=y_test,
    )
    print(f"\n==== 3-VIEW{config.n_outputs}-OUTPUT MODEL RESULTS ====\n")
    for key, value in results.items():
        print(f"{key}: {value}")

    wandb.log(results)


def split_data(
    transcriptomics: pd.DataFrame,
    fluxomics: pd.DataFrame,
    growth: pd.DataFrame,
    proteomics: pd.DataFrame,
    random_state: int,
    test_size: float,
) -> dict[str, pd.DataFrame]:
    """Split the data into training and test sets.

    Parameters
    ----------
    transcriptomics : pd.DataFrame
        The transcriptomics data.
    fluxomics : pd.DataFrame
        The fluxomics data.
    growth : pd.DataFrame
        The growth data.
    proteomics : pd.DataFrame
        The proteomics data.
    random_state : int
        Random seed for reproducibility.
    test_size : float
        The proportion of the dataset to include in the test split.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and test sets.
        keys: "transcriptomics_train", "transcriptomics_test", "fluxomics_train", "fluxomics_test",
    """

    # Generate the split indices using one of the datasets
    split_result = train.random_split(
        data=transcriptomics,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices = split_result["train"]
    test_indices = split_result["test"]

    transcriptomics_data = train.apply_indices_split(
        data=transcriptomics,
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )
    fluxomics_data = train.apply_indices_split(
        data=fluxomics,
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )
    growth_data = train.apply_indices_split(
        data=growth,
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )
    proteomics_data = train.apply_indices_split(
        data=proteomics,
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )

    results = {
        "proteomics_train": proteomics_data["train"],
        "proteomics_test": proteomics_data["test"],
        "transcriptomics_train": transcriptomics_data["train"],
        "transcriptomics_test": transcriptomics_data["test"],
        "fluxomics_train": fluxomics_data["train"],
        "fluxomics_test": fluxomics_data["test"],
        "growth_train": growth_data["train"],
        "growth_test": growth_data["test"],
    }
    return results


if __name__ == "__main__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="growth-bench",
        # track hyperparameters and run metadata with wandb.config
        config=config,
    )
    three_view_moma(config=wandb.config)
