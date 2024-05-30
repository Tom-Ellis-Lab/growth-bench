import random
import sys

import pandas as pd
import pyreadr
from sklearn import preprocessing as sklearn_preprocessing
import tensorflow as tf
import wandb


sys.path.append(".")

from bench.models.moma import model, train, preprocessing
from bench.models.moma.culley_moma import culley_preprocessing


config = {
    "epochs": 1000,
    "neurons": 1000,
    "batch_size": 256,
    "learning_rate": 0.005,
    "momentum": 0.75,
    "optimizer": "sgd",
    "n_outputs": 1,
    "transcriptomics_weights_filename": "gene_expression_weights.h5",
    "fluxomics_weights_filename": "fluxomic_weights.h5",
    "weights_filename": "culley",
}


def culley_main(config: wandb.Config) -> None:
    print("\n==== CULLEY TRANSCRIPTOMICS FLUXOMICS TWO-VIEW MODEL ====\n")
    tf.random.set_seed(42)
    random.seed(42)
    culley_data = get_culley_train_test_data()
    X_train = culley_data["X_train"]
    y_train = culley_data["y_train"]
    X_test = culley_data["X_test"]
    y_test = culley_data["y_test"]

    transcriptomics_train = X_train["transcriptomics"]
    fluxomics_train = X_train["fluxomics"]
    transcriptomics_test = X_test["transcriptomics"]
    fluxomics_test = X_test["fluxomics"]

    scaler = sklearn_preprocessing.StandardScaler().fit(transcriptomics_train)
    transcriptomics_train = scaler.transform(transcriptomics_train)
    transcriptomics_test = scaler.transform(transcriptomics_test)

    scaler = sklearn_preprocessing.StandardScaler().fit(fluxomics_train)
    fluxomics_train = scaler.transform(fluxomics_train)
    fluxomics_test = scaler.transform(fluxomics_test)
    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()

    transcriptomics_model = model.init_single_view_model(
        input_dim=transcriptomics_train.shape[1],
        model_name="transcriptomics",
        input_neurons=config.neurons,
    )
    transcriptomics_model.load_weights(
        f"data/models/moma/{config.transcriptomics_weights_filename}"
    )

    fluxomics_model = model.init_single_view_model(
        input_dim=fluxomics_train.shape[1],
        model_name="fluxomics",
        input_neurons=config.neurons,
    )

    fluxomics_model.load_weights(
        f"data/models/moma/{config.fluxomics_weights_filename}"
    )

    double_view_model = model.init_double_view_model(
        input1_dim=fluxomics_train.shape[1],
        input2_dim=transcriptomics_train.shape[1],
        neurons=config.neurons,
        model_1=fluxomics_model,
        model_2=transcriptomics_model,
    )

    optimiser = preprocessing.get_optimiser(config=config)

    double_view_model.compile(
        loss="mean_squared_error", optimizer=optimiser, metrics=["mean_absolute_error"]
    )

    history = double_view_model.fit(
        x=[fluxomics_train, transcriptomics_train],
        y=y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=([fluxomics_test, transcriptomics_test], y_test),
        verbose="auto",
    )

    double_view_model.save_weights(
        f"data/models/moma/{config.weights_filename}.weights.h5"
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
        name="culley_model_loss",
    )

    results = train.evaluate(
        model=double_view_model,
        X_test=[fluxomics_test, transcriptomics_test],
        y_test=y_test,
    )
    print("\n==== MOMA 2-VIEW MODEL RESULTS ====\n")
    for key, value in results.items():
        print(f"{key}: {value}")

    wandb.log(results)


def get_culley_train_test_data() -> dict[str, pd.DataFrame]:
    """Get the training and testing data for the Culley dataset.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and testing data.
        keys: X_train, y_train, X_test, y_test
    """
    preprocessed_data: dict[str, pd.DataFrame] = get_culley_data()

    random_state = 42
    test_size = 0.2

    # Generate the split indices using one of the datasets
    split_result = train.random_split(
        preprocessed_data["transcriptomics"],
        test_size=test_size,
        random_state=random_state,
    )
    train_indices = split_result["train"]
    test_indices = split_result["test"]

    transcriptomics_data = train.apply_indices_split(
        data=preprocessed_data["transcriptomics"],
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )
    fluxomics_data = train.apply_indices_split(
        data=preprocessed_data["fluxomics"],
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )
    growth_data = train.apply_indices_split(
        data=preprocessed_data["growth"],
        train_indices=train_indices.index,
        test_indices=test_indices.index,
    )

    X_train = {
        "transcriptomics": transcriptomics_data["train"],
        "fluxomics": fluxomics_data["train"],
    }
    y_train = growth_data["train"]
    X_test = {
        "transcriptomics": transcriptomics_data["test"],
        "fluxomics": fluxomics_data["test"],
    }
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


def get_culley_data() -> dict[str, pd.DataFrame]:
    """Get the Culley dataset.

    Returns
    -------
    dict[str, pd.DataFrame]
        The Culley dataset.
        keys: transcriptomics, fluxomics, growth
    """
    print("Loading data...")
    full_data: pd.DataFrame = pyreadr.read_r("data/models/moma/complete_dataset.RDS")[
        None
    ]
    transcriptomics_data: pd.DataFrame = pyreadr.read_r(
        "data/models/moma/gene_expression_dataset.RDS"
    )[None]

    fluxomic_data = full_data.drop(columns=transcriptomics_data.columns.values)
    print("Shape of fluxomics data", fluxomic_data.shape)
    transcriptomics_data["knockout_id"] = full_data["Row"]
    print("Shape of transcriptomics data", transcriptomics_data.shape)
    # ORIGINAL Duibhir growth rates
    growth_data = full_data[["Row", "log2relT"]]

    standard_and_systematic_names = pd.read_csv(
        "bench/models/moma/yeast_gene_names.tsv", sep="\t"
    )
    names_mapping = create_names_mapping(
        standard_and_systematic_names=standard_and_systematic_names
    )
    print("\n==== DONE ====\n")
    print("Preprocessing data...")
    result: dict[str, pd.DataFrame] = culley_preprocessing.culley_preprocessing(
        transcriptomics_data=transcriptomics_data,
        fluxomics_data=fluxomic_data,
        growth_data=growth_data,
        mapping_dict=names_mapping,
    )
    return result


def create_names_mapping(
    standard_and_systematic_names: pd.DataFrame,
) -> dict[str, str]:
    """Create a mapping between standard and systematic names

    Parameters
    ----------
    standard_and_systematic_names : pd.DataFrame
        The mapping between standard and systematic names
    Returns
    -------
    pd.DataFrame
        The data with the standard names converted to systematic names.
    """
    result = pd.Series(
        standard_and_systematic_names["systematic_name"].values,
        index=standard_and_systematic_names["standard_name"],
    ).to_dict()
    # Also add systematic names as keys mapping to themselves
    result.update(
        pd.Series(
            standard_and_systematic_names["systematic_name"].values,
            index=standard_and_systematic_names["systematic_name"],
        ).to_dict()
    )
    return result


if __name__ == "__main__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="growth-bench",
        # track hyperparameters and run metadata with wandb.config
        config=config,
    )
    culley_main(config=wandb.config)
