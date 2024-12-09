import keras
import pytest

from bench.models.moma import model_constructors


def test_layer_type():
    # Check that the enum has the expected values
    expected_layers = {"Dense", "Dropout", "Conv2D", "Flatten"}
    actual_layers = {layer.value for layer in model_constructors.LayerType}
    assert actual_layers == expected_layers


def test_layer_config():
    layer_config = model_constructors.LayerConfig(
        type=model_constructors.LayerType.DENSE, params={"units": 32}
    )

    # Check that the dataclass has the expected values
    assert layer_config.type == model_constructors.LayerType.DENSE
    assert layer_config.params == {"units": 32}

    # Check that the dataclass has the expected attributes
    expected_attributes = {"type", "params"}
    actual_attributes = set(vars(layer_config).keys())
    assert actual_attributes == expected_attributes


def test_model_params():
    model_params = model_constructors.ModelParams(
        name="model",
        architecture=[
            model_constructors.LayerConfig(
                type=model_constructors.LayerType.DENSE, params={"units": 32}
            )
        ],
        input_size=10,
    )

    # Check that the dataclass has the expected values
    assert model_params.name == "model"
    assert model_params.architecture == [
        model_constructors.LayerConfig(
            type=model_constructors.LayerType.DENSE, params={"units": 32}
        )
    ]
    assert model_params.input_size == 10

    # Check that the dataclass has the expected attributes
    expected_attributes = {"name", "architecture", "input_size"}
    actual_attributes = set(vars(model_params).keys())
    assert actual_attributes == expected_attributes


@pytest.mark.parametrize(
    "data",
    [
        {
            "id": "dense_layer",
            "input": model_constructors.LayerConfig(
                type=model_constructors.LayerType.DENSE, params={"units": 32}
            ),
            "expected": keras.layers.Dense(units=32),
        },
        {
            "id": "dropout_layer",
            "input": model_constructors.LayerConfig(
                type=model_constructors.LayerType.DROPOUT, params={"rate": 0.2}
            ),
            "expected": keras.layers.Dropout(rate=0.2),
        },
        {
            "id": "conv2d_layer",
            "input": model_constructors.LayerConfig(
                type=model_constructors.LayerType.CONV2D,
                params={"filters": 32, "kernel_size": (3, 3)},
            ),
            "expected": keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
        },
        {
            "id": "flatten_layer",
            "input": model_constructors.LayerConfig(
                type=model_constructors.LayerType.FLATTEN, params={}
            ),
            "expected": keras.layers.Flatten(),
        },
    ],
    ids=lambda case: case["id"],
)
class TestLayerFactory:

    def test_create_layer(self, data):
        layer = model_constructors.LayerFactory.create_layer(data["input"])
        assert isinstance(layer, type(data["expected"]))


class TestModelConstructor:

    def test_construct(self):
        model_params = model_constructors.ModelParams(
            name="model",
            architecture=[
                model_constructors.LayerConfig(
                    type=model_constructors.LayerType.DENSE, params={"units": 32}
                )
            ],
            input_size=10,
        )

        observed = model_constructors.SingleViewModelConstructor().construct(
            model_params=model_params
        )

        assert isinstance(observed, keras.Model)
        assert len(observed.layers) == 2
        assert isinstance(observed.layers[0], keras.layers.InputLayer)
        assert isinstance(observed.layers[1], keras.layers.Dense)
        assert observed.layers[1].units == 32
        assert observed.name == "model"
        assert observed.input_shape == (None, 10)


class TestMultiViewModelParams:

    @pytest.fixture
    def model1(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(10,)),
                keras.layers.Dense(units=32),
            ]
        )

    @pytest.fixture
    def model2(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer(input_shape=(20,)),
                keras.layers.Dense(units=32),
            ]
        )

    def test_init(self, model1, model2):

        model_params = model_constructors.MultViewModelParams(
            name="model",
            architecture=[
                model_constructors.LayerConfig(
                    type=model_constructors.LayerType.DENSE, params={"units": 32}
                )
            ],
            models={10: model1, 20: model2},
        )

        # Check that the dataclass has the expected values
        assert model_params.name == "model"
        assert model_params.architecture == [
            model_constructors.LayerConfig(
                type=model_constructors.LayerType.DENSE, params={"units": 32}
            )
        ]
        assert model_params.models == {10: model1, 20: model2}

        # Check that the dataclass has the expected attributes
        expected_attributes = {"name", "architecture", "models"}
        actual_attributes = set(vars(model_params).keys())
        assert actual_attributes == expected_attributes

    def test_wonrg_input_size(self, model1, model2):
        with pytest.raises(ValueError):
            model_params = model_constructors.MultViewModelParams(
                name="model",
                architecture=[
                    model_constructors.LayerConfig(
                        type=model_constructors.LayerType.DENSE, params={"units": 32}
                    )
                ],
                models={15: model1, 25: model2},
            )


class TestMultiViewModelConstructor:

    @pytest.fixture
    def models(self):
        model1_params = model_constructors.ModelParams(
            name="model1",
            architecture=[
                model_constructors.LayerConfig(
                    type=model_constructors.LayerType.DENSE, params={"units": 32}
                )
            ],
            input_size=10,
        )

        model2_params = model_constructors.ModelParams(
            name="model2",
            architecture=[
                model_constructors.LayerConfig(
                    type=model_constructors.LayerType.DENSE, params={"units": 32}
                )
            ],
            input_size=20,
        )

        model1 = model_constructors.SingleViewModelConstructor().construct(
            model_params=model1_params
        )
        model2 = model_constructors.SingleViewModelConstructor().construct(
            model_params=model2_params
        )

        return {10: model1, 20: model2}

    def test_construct(self, models):
        model_params = model_constructors.MultViewModelParams(
            name="model",
            architecture=[
                model_constructors.LayerConfig(
                    type=model_constructors.LayerType.DENSE, params={"units": 32}
                )
            ],
            models=models,
        )

        observed = model_constructors.MultiViewModelConstructor().construct(
            model_params=model_params
        )

        print(observed.layers)

        assert isinstance(observed, keras.Model)
        assert len(observed.layers) == 6
        assert isinstance(observed.layers[0], keras.layers.InputLayer)
        assert isinstance(observed.layers[1], keras.layers.InputLayer)
        assert isinstance(observed.layers[2], keras.Model)
        assert isinstance(observed.layers[3], keras.Model)
        assert isinstance(observed.layers[4], keras.layers.Concatenate)
        assert isinstance(observed.layers[5], keras.layers.Dense)
        assert observed.layers[5].units == 32
        assert observed.name == "model"
        assert observed.input_shape == [(None, 10), (None, 20)]
