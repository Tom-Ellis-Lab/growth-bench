import csv
import sys

import sklearn

import pandas as pd

from sklearn import preprocessing

sys.path.append(".")

from bench.models.moma import model
from bench.models.moma.ralser_moma import ralser_preprocessing, ralser_train

LEARNING_RATE = 0.005
EPOCHS = 1000
MOMENTUM = 0.75
NEURONS = 1000
BATCHES = 256
VALIDATION = 0.1

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

    proteomics_data = ralser_train.split(data=proteomics_data, test_indices=test_indices)
    growth_data = ralser_train.split(data=growth_data, test_indices=test_indices)

    X_train = proteomics_data["train"]
    y_train = growth_data["train"]
    X_test = proteomics_data["test"]
    y_test = growth_data["test"]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n==== BUILDING NETWORK ====\n")
    proteomics_model = model.init_single_view_model(
        input_dim=X_train.shape[1],
        model_name="proteomics",
        neurons=NEURONS,
    )
    print("\n==== DONE ====\n")

    print("\n==== TRAINING ====\n")
    history = ralser_train.train_model(
        model=proteomics_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batches=BATCHES,
        momentum=MOMENTUM,
        validation=VALIDATION,
        weights_to_save_dir="data/models/moma/",
        weights_name="",
    )
    print("\n==== DONE ====\n")

    ralser_train._plot_loss(history=history, plot_to_save_dir="data/models/moma/")

    results = ralser_train._evaluate(
        model=proteomics_model,
        X_test=X_test,
        y_test=y_test.to_numpy(),
    )
    for key, value in results.items():
        print(f"{key}: {value}")



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