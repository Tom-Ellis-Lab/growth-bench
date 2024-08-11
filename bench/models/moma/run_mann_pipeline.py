import random
import sys

import tensorflow as tf
import wandb

sys.path.append(".")

from bench.models.moma import model, preprocessing, train, view


# Configuration dictionary for training the Multiomics Artificial Neural Network (MANN) (below).
# This configuration includes hyperparameters, model weights, and settings for data processing.
# It is used to control various aspects of the training process, including the number of epochs,
# learning rate, and input types, allowing for flexible experimentation.

FLUXOMICS_WEIGHTS = "fluxomics_growthralser_lr0.005_batch256_neurons500_optimiseradagrad_output1.weights.h5"
TRANSCRIPTOMICS_WEIGHTS = "transcriptomics_growthralser_lr0.005_batch128_neurons100_optimisersgd_output1.weights.h5"
PROTEOMICS_WEIGHTS = "proteomics_growthralser_lr0.06_batch128_neurons100_optimiseradagrad_output1.weights.h5"

config = {
    "epochs": 10,
    "neurons": {
        "proteomics": 100,
        "transcriptomics": 100,
        "fluxomics": 500,
        "multimodal": 500,
    },
    "batch_size": 64,
    "learning_rate": 0.007,
    "momentum": 0.75,
    "optimizer": "sgd",  # "sgd", "adam", "adagrad"
    "medium": "SC",  # "SC", "SM", "YPD"
    "dropout": {
        "proteomics": 0.1,
        "transcriptomics": 0.1,
        "fluxomics": 0.1,
        "multimodal": 0.1,
    },
    "input_type": [
        "fluxomics",
        "transcriptomics",
        "proteomics",
    ],
    "models_weights": {  # specify the weights file for each input type that you want to use
        "proteomics": PROTEOMICS_WEIGHTS,
        "transcriptomics": TRANSCRIPTOMICS_WEIGHTS,
        "fluxomics": FLUXOMICS_WEIGHTS,
    },
    "save_weights": False,
    "set_seed": True,
    "cross_validation": None,  # if yes, choose # fold (for example 5), otherwise: None
}


def main(config: wandb.Config) -> None:
    """Main function to train and evaluate the MANN model.

    MANN = Multiomics Artificial Neural Network

    This function orchestrates the following steps in the pipeline:

    1. **Configuration Verification**: Validates that the configuration parameters are correct and consistent.
    2. **Logging Configuration**: Logs the configuration settings for transparency.
    3. **Data Gathering**: Fetches the necessary multiomics data based on the specified input types and medium.
    4. **Data Filtering**: Filters the acquired data to remove irrelevant or low-quality data points.
    5. **Data Splitting**: Splits the filtered data into training, validation, and testing sets. Handles cross-validation if specified.
    6. **Model Training**: Builds the Multiomics Artificial Neural Network (MANN) model and trains it on the training data.
    7. **Evaluation**: Evaluates the model on the validation/test data, capturing key metrics.
    8. **Loss Plotting**: Plots and logs the training and validation loss curves to visualize model performance.
    9. **Results Logging**: Logs the results for each fold (if cross-validation is used) and the overall results.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.
    """
    print("\n==== START ====\n")
    print(f"{len(config.input_type)}-VIEW 1-OUTPUT MANN MODEL")
    print("=====================================")

    # 1. **Configuration Verification**
    verify_config_parameters(config=config)
    # 2. **Logging Configuration**
    view.log_configuations(config=config)

    if config.set_seed:
        set_seed(seed=42)

    # 3. **Data Gathering**
    total_data = preprocessing.get_data(
        medium=config.medium, input_type=config.input_type
    )
    # 4. **Data Filtering**
    filtered_data = preprocessing.filter_data(datasets=total_data)

    # 5. **Data Splitting**
    data = preprocessing.split_data(
        data=filtered_data,
        set_seed=config.set_seed,
        cross_validation=config.cross_validation,
    )

    # 6. **Model Training**
    folds_results = {}
    for fold in data.keys():

        fold_data = data[fold]

        scaled_data = preprocessing.scale_data(data=fold_data)

        mann_model = model.build_multiview_model(
            config=config,
            train_data=scaled_data["scaled_train"],
            input_neurons=config.neurons["multimodal"],
        )

        # 7. **Evaluation**
        result = train.train_and_evaluate(
            config=config,
            model=mann_model,
            data=scaled_data,
        )

        # 8. **Loss Plotting**
        history = result[config.medium].pop("history")
        if not isinstance(history, dict):
            raise ValueError("The history is not a dictionary.")

        view.plot_model_loss(
            data=scaled_data, history=history, config=config, fold=fold
        )

        view.log_single_fold_results(result=result[config.medium], fold=fold)
        folds_results[fold] = result

    # 9. **Results Logging**
    view.log_results(validation_results=folds_results, config=config)


def verify_config_parameters(config: wandb.Config) -> None:
    """Verify the configuration parameters.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.
    """
    if len(config.input_type) != len(config.neurons) - 1 or len(
        config.input_type
    ) != len(config.models_weights):
        raise ValueError(
            "The number of input types must be equal to the number of neurons and models weights."
        )


def set_seed(seed: int) -> None:
    """Set the seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value.
    """
    tf.random.set_seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # Initialize Weights & Biases (wandb) for experiment tracking and visualization.
    # wandb helps log hyperparameters, system metrics, and outputs, enabling easy comparison
    # of different runs and facilitating reproducibility of experiments.
    wandb.init(
        # set the wandb project where this run will be logged
        project="growth-bench",
        # track hyperparameters and run metadata with wandb.config
        config=config,
    )
    main(config=wandb.config)
    print("\n==== DONE ====\n")
