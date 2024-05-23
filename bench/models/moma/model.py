import tensorflow as tf


def init_single_view_model(
    input_dim: int,
    model_name: str,
    input_neurons: int,
    output_neurons: int = 1,
) -> tf.keras.Model:
    """Initialize a model with the given parameters.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    model_name : str
        The name of the model. Used for naming the layers.

    Returns
    -------
    tf.keras.Model
    """
    # Input layer
    input = tf.keras.layers.Input(shape=(input_dim,))

    # Hidden layer (1)
    layer = tf.keras.layers.Dense(
        input_neurons,
        activation="sigmoid",
        kernel_constraint=tf.keras.constraints.max_norm(3),
        name=f"{model_name}_1",
    )(input)
    # Set 40% of input units to 0 at each update during training time
    layer = tf.keras.layers.Dropout(rate=0.4)(layer)

    # Hidden layer (2)
    layer = tf.keras.layers.Dense(
        input_neurons,
        activation="sigmoid",
        kernel_constraint=tf.keras.constraints.max_norm(3),
        name=f"{model_name}_2",
    )(layer)
    # Set 40% of input units to 0 at each update during training time
    layer = tf.keras.layers.Dropout(rate=0.4)(layer)

    # Final output layer
    predictions = tf.keras.layers.Dense(output_neurons, activation="linear")(layer)
    model = tf.keras.Model(inputs=input, outputs=predictions)
    print(f"Summary of the single-view model {model_name}")
    model.summary()
    return model


def init_double_view_model(
    input1_dim: int,
    input2_dim: int,
    neurons: int,
    model_1: tf.keras.Model,
    model_2: tf.keras.Model,
) -> tf.keras.Model:
    """Initialize a model with two inputs and one output.

    Parameters
    ----------
    input1_dim : int
        The number of features in the first input.
    input2_dim : int
        The number of features in the second input.
    neurons : int
        The number of neurons in the hidden layers.
    model_1 : tf.keras.Model
        The first model to use.
    model_2 : tf.keras.Model
        The second model to use.

    Returns
    -------
    tf.keras.Model

    """
    input_1 = tf.keras.layers.Input(shape=(input1_dim,))
    input_2 = tf.keras.layers.Input(shape=(input2_dim,))

    combined_layer = tf.keras.layers.Concatenate()([model_1(input_1), model_2(input_2)])
    combined_layer = tf.keras.layers.Dense(
        neurons,
        activation="sigmoid",
        kernel_constraint=tf.keras.constraints.max_norm(3),
        name="last_hidden",
    )(combined_layer)

    predictions = tf.keras.layers.Dense(1, activation="linear")(combined_layer)
    result = tf.keras.Model(inputs=[input_1, input_2], outputs=predictions)
    return result


def init_triple_view_model(
    input1_dim: int,
    input2_dim: int,
    input3_dim: int,
    neurons: int,
    model_1: tf.keras.Model,
    model_2: tf.keras.Model,
    model_3: tf.keras.Model,
) -> tf.keras.Model:
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
    model_1 : tf.keras.Model
        The first model to use.
    model_2 : tf.keras.Model
        The second model to use.
    model_3 : tf.keras.Model
        The third model to use.

    Returns
    -------
    tf.keras.Model

    """
    input_1 = tf.keras.layers.Input(shape=(input1_dim,))
    input_2 = tf.keras.layers.Input(shape=(input2_dim,))
    input_3 = tf.keras.layers.Input(shape=(input3_dim,))

    combined_layer = tf.keras.layers.Concatenate()(
        [model_1(input_1), model_2(input_2), model_3(input_3)]
    )
    combined_layer = tf.keras.layers.Dense(
        neurons,
        activation="sigmoid",
        kernel_constraint=tf.keras.constraints.max_norm(3),
        name="last_hidden",
    )(combined_layer)

    predictions = tf.keras.layers.Dense(1, activation="linear")(combined_layer)
    result = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=predictions)
    return result
