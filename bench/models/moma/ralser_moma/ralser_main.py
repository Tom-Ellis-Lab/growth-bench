import csv
import sys

import pandas as pd
from sklearn import preprocessing
import wandb
from wandb.integration.keras import WandbMetricsLogger

sys.path.append(".")

from bench.models.moma import model, train
from bench.models.moma.ralser_moma import ralser_preprocessing, ralser_train

wandb.init(
    # set the wandb project where this run will be logged
    project="growth-bench",
    # track hyperparameters and run metadata with wandb.config
    config={
        "epochs": 1000,
        "neurons": 1000,
        "batch_size": 256,
    },
)

config = wandb.config


def ralser_main():
    print("\n==== RALSER PROTEOMICS SINGLE VIEW MODEL ====\n")
    print("Loading data...")
    proteomics_data_ralser = pd.read_csv("data/models/moma/yeast5k_impute_wide.csv")
    print("Shape of proteome data", proteomics_data_ralser.shape)
    growth_rates_ralser = pd.read_csv("data/tasks/task3/yeast5k_growthrates_byORF.csv")
    print("Shape of growth data", growth_rates_ralser.shape)
    print("\n==== DONE ====\n")
    print("Preprocessing data...")
    preprocessed_data = ralser_preprocessing.ralser_preprocessing(
        proteomics_data=proteomics_data_ralser,
        growth_data=growth_rates_ralser[["orf", "SC"]],
    )
    print("\n==== DONE ====\n")
    test_indices = _get_test_indices()

    proteomics_data = preprocessed_data["proteomics"]
    growth_data = preprocessed_data["growth"]

    proteomics_data = ralser_train.split(
        data=proteomics_data, test_indices=test_indices
    )
    growth_data = ralser_train.split(data=growth_data, test_indices=test_indices)

    X_train = proteomics_data["train"]
    y_train = growth_data["train"]
    X_test = proteomics_data["test"]
    y_test = growth_data["test"]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)  # TODO: make sure this is of type np.ndarray
    X_test = scaler.transform(X_test)
    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()

    print("\n==== BUILDING NETWORK ====\n")
    proteomics_model = model.init_single_view_model(
        input_dim=X_train.shape[1],
        model_name="proteomics",
        neurons=config.neurons,
    )
    print("\n==== DONE ====\n")

    print("\n==== TRAINING ====\n")
    history = ralser_train.train_model(
        model=proteomics_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=config.epochs,
        batch_size=config.batch_size,
        weights_to_save_dir="data/models/moma/",
        weights_name="",
        callbacks=[
            WandbMetricsLogger(),
        ],
    )
    print("\n==== DONE ====\n")

    total_samples = len(X_train)
    normalized_loss = [
        loss * config.batch_size / total_samples for loss in history.history["loss"]
    ]
    normalized_val_loss = [
        val_loss * config.batch_size / len(X_test)
        for val_loss in history.history["val_loss"]
    ]
    ralser_train._plot_loss(
        loss=normalized_loss,
        val_loss=normalized_val_loss,
        plot_to_save_dir="data/models/moma/",
        name="proteomics",
    )

    results = ralser_train._evaluate(
        model=proteomics_model,
        X_test=X_test,
        y_test=y_test,
    )
    print("\n==== RALSER 1-VIEW PROTEOMICS MODEL RESULTS ====\n")
    for key, value in results.items():
        print(f"{key}: {value}")

    wandb.log(results)


def get_ralser_train_test_data(
    train_indices: pd.Index, test_indices: pd.Index
) -> dict[str, pd.DataFrame]:
    """Get the training and test sets for the Ralser proteomics model.

    Parameters
    ----------
    train_indices : pd.Index
        The training set indices.
    test_indices : pd.Index
        The test set indices.

    Returns
    -------
    dict[str, pd.DataFrame]
        The training and test sets.
    """
    proteomics_data_ralser = pd.read_csv("data/models/moma/yeast5k_impute_wide.csv")
    growth_rates_ralser = pd.read_csv("data/tasks/task3/yeast5k_growthrates_byORF.csv")

    preprocessed_data = ralser_preprocessing.ralser_preprocessing(
        proteomics_data=proteomics_data_ralser,
        growth_data=growth_rates_ralser[["orf", "SC"]],
    )

    proteomics_data = preprocessed_data["proteomics"]
    growth_data = preprocessed_data["growth"]

    proteomics_data = train.apply_indices_split(
        data=proteomics_data, train_indices=train_indices, test_indices=test_indices
    )

    growth_data = train.apply_indices_split(
        data=growth_data, train_indices=train_indices, test_indices=test_indices
    )

    return {
        "X_train": proteomics_data["train"],
        "y_train": growth_data["train"],
        "X_test": proteomics_data["test"],
        "y_test": growth_data["test"],
    }


def _get_test_indices() -> list[int]:
    """Get the test indices from the test_indices_ralser.csv file.

    Returns
    -------
    list[int]
        The test indices.
    """
    with open("data/models/moma/test_indices_ralser.csv", "r") as csvfile:
        test_indices = []
        for row in csv.reader(csvfile, delimiter=";"):
            test_indices.append(row[0])
    test_indices = list(map(int, test_indices))
    return test_indices


if __name__ == "__main__":
    ralser_main()
