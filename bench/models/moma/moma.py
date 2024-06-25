import random
import sys

from sklearn.model_selection import KFold
import tensorflow as tf
import wandb


sys.path.append(".")

from bench.models.moma import model, preprocessing, train


config = {
    "epochs": 1000,
    "neurons": {
        "proteomics": 500,
        "transcriptomics": 500,
        "fluxomics": 500,
        "multimodal": 500,
    },
    "batch_size": 256,
    "learning_rate": 0.005,
    "momentum": 0.75,
    "optimizer": "adam",  # "sgd", "adam", "adagrad"
    "medium": "SC",  # "SC", "SM", "YPD"
    "dropout": {
        "proteomics": 0.4,
        "transcriptomics": 0.4,
        "fluxomics": 0.4,
        "multimodal": 0.4,
    },
    "input_type": [
        "fluxomics",
        "transcriptomics",
        "proteomics",
    ],
    "models_weights": {  # specify the weights file for each input type that you want to use
        "proteomics": ".weights.h5",
        "transcriptomics": ".weights.h5",
        "fluxomics": ".weights.h5",
    },
    "save_weights": False,
    "set_seed": True,
    "cross_validation": None,  # if yes, choose # fold (for example 5), otherwise: None
    "n_selected_genes": 5,
}


def main(config: wandb.Config) -> None:
    """Main function to train and evaluate the MOMA model.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.
    """
    print("\n==== START ====\n")
    print(f"{len(config.input_type)}-VIEW 1-OUTPUT MOMA MODEL")
    print("=====================================")

    if len(config.input_type) != len(config.neurons) - 1 or len(
        config.input_type
    ) != len(config.models_weights):
        raise ValueError(
            "The number of input types must be equal to the number of neurons and models weights."
        )
    log_configuations(config=config)
    if config.set_seed:
        tf.random.set_seed(42)
        random.seed(42)

    total_data = preprocessing.get_data(
        medium=config.medium, input_type=config.input_type
    )
    filtered_data = preprocessing.filter_data(datasets=total_data)

    if config.cross_validation is not None:
        kf = KFold(n_splits=config.cross_validation, shuffle=True, random_state=42)
        folds_results = {}
        for fold, (train_index, test_index) in enumerate(
            kf.split(filtered_data["growth"])
        ):
            data = train.split_data_using_indices(
                data=filtered_data, train_index=train_index, test_index=test_index
            )

            data = preprocessing.scale_data(data=data)

            models = model.build_models(
                config=config,
                train_data=data["scaled_train"],
            )
            final_model = model.concatenate_model_into_multiview(
                models=models,
                input_neurons=config.neurons["multimodal"],
                data=data["scaled_train"],
            )

            results = train.train_and_evaluate(
                config=config,
                model=final_model,
                data=data,
            )
            history = results[config.medium].pop("history")
            if not isinstance(history, dict):
                raise ValueError("The history is not a dictionary.")

            normalized_loss = preprocessing.get_normalised_loss(
                data=data,
                history=history,
                config=config,
                type_of_loss="loss",
            )

            normalized_val_loss = preprocessing.get_normalised_loss(
                data=data,
                history=history,
                config=config,
                type_of_loss="val_loss",
            )
            train.plot_loss(
                loss=normalized_loss,
                val_loss=normalized_val_loss,
                name=f"fold_{fold+1}",
            )
            for key, value in results.items():
                print(f"FOLD {fold+1} - {key}: {value}")
            print(f"DONE: Training on fold {fold+1}/5 completed")

            folds_results[f"fold_{fold+1}"] = results
        log_cross_validation_results(config=config, folds_results=folds_results)

    else:

        data = train.split_data_using_names(
            data=filtered_data, set_seed=config.set_seed
        )

        data = preprocessing.scale_data(data=data)

        models = model.build_models(
            config=config,
            train_data=data["scaled_train"],
        )
        final_model = model.concatenate_model_into_multiview(
            models=models,
            input_neurons=config.neurons["multimodal"],
            data=data["scaled_train"],
        )

        results = train.train_and_evaluate(
            config=config,
            model=final_model,
            data=data,
        )

        ####
        growth_test = data["test"]["growth"]
        if len(config.input_type) == 1 and config.input_type == "proteomics":
            selected_genes = ["YDR432W", "YDR159W", "YML103C", "YBR274W", "YNL128W"]
        else:
            selected_genes = preprocessing.select_genes_for_analysis(
                growth_rate=growth_test,
                medium=config.medium,
                n_intervals=config.n_selected_genes,
            )
        train.compute_error_on_selected_genes(
            selected_genes=selected_genes,
            data=data,
            trained_model=final_model,
            input_type=config.input_type,
            medium=config.medium,
        )

        history = results[config.medium].pop("history")
        if not isinstance(history, dict):
            raise ValueError("The history is not a dictionary.")
        normalized_loss = preprocessing.get_normalised_loss(
            data=data,
            history=history,
            config=config,
            type_of_loss="loss",
        )

        normalized_val_loss = preprocessing.get_normalised_loss(
            data=data,
            history=history,
            config=config,
            type_of_loss="val_loss",
        )
        if len(config.input_type) == 1:
            name = config.input_type[0]
        else:
            name = "multimodal"
        train.plot_loss(
            loss=normalized_loss,
            val_loss=normalized_val_loss,
            name=f"{name}",
        )
        log_results(results=results, config=config)


def log_configuations(config: wandb.Config) -> None:
    """Log the configurations to wandb.

    Parameters
    ----------

    config : wandb.Config
        The configuration object.

    """
    print("Configuration Table")
    print("-" * 60)
    print(f"{'Hyperparameter':<15}{'Value':<15}")
    print("-" * 60)
    print(f"{'epochs':<15}{config.epochs:<15}")
    print(
        f"{'neurons':<15}{', '.join(f'{k}: {v}' for k, v in config.neurons.items()):<15}"
    )
    print(f"{'batch size':<15}{config.batch_size:<15}")
    print(f"{'learning rate':<15}{config.learning_rate:<15}")
    print(f"{'optimizer':<15}{config.optimizer:<15}")
    print(f"{'momentum':<15}{config.momentum:<15}")
    print(
        f"{'dropout':<15}{', '.join(f'{k}: {v}' for k, v in config.dropout.items()):<15}"
    )
    print(f"{'medium':<15}{config.medium:<15}")
    print(f"{'modalities':<15}{' '.join(str(i) for i in config.input_type):<15}")
    print(f"{'set seed':<15}{str(bool(config.set_seed)):<15}")
    print("-" * 60)


def log_results(results: dict[str, dict[str, float]], config: wandb.Config) -> None:
    """Log the results to wandb.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        The results to log.
    """
    print("=====================================")
    print("TRAINING AND VALIDATION DONE")
    log_configuations(config=config)
    print("=====================================")
    print("Results (to copy)")
    print(results)
    print("=====================================")
    print("Results")
    for key, value in results.items():
        print(f"{key}: {value}")
    wandb.log(results)


def log_cross_validation_results(
    config: wandb.Config, folds_results: dict[str, dict[str, float]]
) -> None:
    """Log the results of the cross-validation.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.

    folds_results : dict[str, dict[str, float]]
        The results of the cross-validation.
    """
    print("=====================================")
    print("CROSS-VALIDATION DONE")
    log_configuations(config=config)
    print("=====================================")
    print("Results (to copy)")
    print(folds_results)
    print("=====================================")
    print("Folds Results")
    for fold, results in folds_results.items():
        print(f"Fold {fold}: {results}")
    wandb.log(folds_results)


def moma_model():
    pass


if __name__ == "__main__":
    wandb.init(
        # set the wandb project where this run will be logged
        project="growth-bench",
        # track hyperparameters and run metadata with wandb.config
        config=config,
    )
    main(config=wandb.config)
    print("\n==== DONE ====\n")
