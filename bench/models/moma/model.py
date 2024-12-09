import keras
import pandas as pd

import wandb


def build_multiview_model(
    config: wandb.Config,
    train_data: dict[str, pd.DataFrame],
    input_neurons: int,
) -> keras.Model:
    """Build the final model for the MOMA model.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.

    train_data : dict[str, pd.DataFrame]
        The training data.

    input_neurons : int
        The number of neurons in the input layer.

    Returns
    -------
    keras.Model
        The final model.
    """
    models = build_models(config=config, train_data=train_data)
    result = concatenate_model_into_multiview(
        models=models, input_neurons=input_neurons, data=train_data
    )
    return result


def build_models(
    config: wandb.Config,
    train_data: dict[str, pd.DataFrame],
) -> dict[str, keras.Model]:
    """Build the models for the MOMA model.

    Parameters
    ----------
    config : wandb.Config
        The configuration object.

    train_data : dict[str, pd.DataFrame]
        The training data.

    Returns
    -------
    dict[str, keras.Model]
        The models.
    """
    models = {}
    for input_type, input_data in train_data.items():
        if input_type != "growth":
            single_view_model = init_single_view_model(
                input_dim=input_data.shape[1],
                model_name=input_type,
                input_neurons=config.neurons[input_type],
                drouput_rate=config.dropout[input_type],
            )
            if len(config.input_type) > 1:
                single_view_model.load_weights(
                    f"data/models/moma/{config.models_weights[input_type]}"
                )
            models[input_type] = single_view_model

    return models


def concatenate_model_into_multiview(
    models: dict[str, keras.Model], input_neurons: int, data: dict[str, pd.DataFrame]
) -> keras.Model:
    """Get the final model by combining the single view models.

    Parameters
    ----------
    models : dict[str, keras.Model]
        The single view models.

    input_neurons : int
        The number of neurons in the input layer.

    data : dict[str, pd.DataFrame]
        The data used to determine the input dimensions.

    Returns
    -------
    keras.Model
        The final model.
    """

    models_list = list(models.values())
    data_list = list(data.values())

    if len(models) == 1:
        result = next(iter(models.values()))
    elif len(models) == 2:
        model_1 = models_list[0]
        model_2 = models_list[1]
        data_1 = data_list[0]
        data_2 = data_list[1]

        result = init_double_view_model(
            model_1=model_1,
            model_2=model_2,
            input1_dim=data_1.shape[1],
            input2_dim=data_2.shape[1],
            neurons=input_neurons,
        )

    elif len(models) == 3:
        model_1 = models_list[0]
        model_2 = models_list[1]
        model_3 = models_list[2]

        data_1 = data_list[0]
        data_2 = data_list[1]
        data_3 = data_list[2]

        result = init_triple_view_model(
            model_1=model_1,
            model_2=model_2,
            model_3=model_3,
            input1_dim=data_1.shape[1],
            input2_dim=data_2.shape[1],
            input3_dim=data_3.shape[1],
            neurons=input_neurons,
        )

    return result


def init_single_view_model(
    input_dim: int,
    model_name: str,
    input_neurons: int,
    drouput_rate: float = 0.4,
    output_neurons: int = 1,
) -> keras.Model:
    """Initialize a model with the given parameters.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    model_name : str
        The name of the model. Used for naming the layers.

    Returns
    -------
    keras.Model
    """
    # Input layer
    input = keras.layers.Input(shape=(input_dim,))

    # Hidden layer (1)
    layer = keras.layers.Dense(
        input_neurons,
        activation="sigmoid",
        kernel_constraint=keras.constraints.max_norm(3),
        name=f"{model_name}_1",
    )(input)
    # Set 40% of input units to 0 at each update during training time
    layer = keras.layers.Dropout(rate=drouput_rate)(layer)

    # Hidden layer (2)
    layer = keras.layers.Dense(
        input_neurons,
        activation="sigmoid",
        kernel_constraint=keras.constraints.max_norm(3),
        name=f"{model_name}_2",
    )(layer)
    # Set 40% of input units to 0 at each update during training time
    layer = keras.layers.Dropout(rate=drouput_rate)(layer)

    # Final output layer
    predictions = keras.layers.Dense(output_neurons, activation="linear")(layer)
    model = keras.Model(inputs=input, outputs=predictions, name=model_name)
    print(f"Summary of the single-view model {model_name}")
    model.summary()
    return model


def init_double_view_model(
    input1_dim: int,
    input2_dim: int,
    neurons: int,
    model_1: keras.Model,
    model_2: keras.Model,
) -> keras.Model:
    """Initialize a model with two inputs and one output.

    Parameters
    ----------
    input1_dim : int
        The number of features in the first input.
    input2_dim : int
        The number of features in the second input.
    neurons : int
        The number of neurons in the hidden layers.
    model_1 : keras.Model
        The first model to use.
    model_2 : keras.Model
        The second model to use.

    Returns
    -------
    keras.Model

    """
    input_1 = keras.layers.Input(shape=(input1_dim,))
    input_2 = keras.layers.Input(shape=(input2_dim,))

    combined_layer = keras.layers.Concatenate()([model_1(input_1), model_2(input_2)])
    combined_layer = keras.layers.Dense(
        neurons,
        activation="sigmoid",
        kernel_constraint=keras.constraints.max_norm(3),
        name="last_hidden",
    )(combined_layer)

    predictions = keras.layers.Dense(1, activation="linear")(combined_layer)
    result = keras.Model(
        inputs=[input_1, input_2], outputs=predictions, name="double_view"
    )
    return result


def init_triple_view_model(
    input1_dim: int,
    input2_dim: int,
    input3_dim: int,
    neurons: int,
    model_1: keras.Model,
    model_2: keras.Model,
    model_3: keras.Model,
    output_neurons: int = 1,
) -> keras.Model:
    """Initialize a model with two inputs and one output.

    Parameters
    ----------
    input1_dim : int
        The number of features in the first input.
    input2_dim : int
        The number of features in the second input.
    input3_dim : int
        The number of features in the third input.
    neurons : int
        The number of neurons in the hidden layers.
    model_1 : keras.Model
        The first model to use.
    model_2 : keras.Model
        The second model to use.
    model_3 : keras.Model
        The third model to use.

    Returns
    -------
    keras.Model

    """
    input_1 = keras.layers.Input(shape=(input1_dim,))
    input_2 = keras.layers.Input(shape=(input2_dim,))
    input_3 = keras.layers.Input(shape=(input3_dim,))

    combined_layer = keras.layers.Concatenate()(
        [model_1(input_1), model_2(input_2), model_3(input_3)]
    )
    combined_layer = keras.layers.Dense(
        neurons,
        activation="sigmoid",
        kernel_constraint=keras.constraints.max_norm(3),
        name="last_hidden",
    )(combined_layer)

    predictions = keras.layers.Dense(output_neurons, activation="linear")(
        combined_layer
    )
    result = keras.Model(
        inputs=[input_1, input_2, input_3], outputs=predictions, name="triple_view"
    )
    return result
