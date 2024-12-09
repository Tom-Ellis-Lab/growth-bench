import keras
import pytest

from bench.models.moma import optimiser_factory


def test_optimiser_type():
    expected_optimisers = {"Adam", "SGD", "Adagrad"}
    actual_optimisers = {
        optimiser.value for optimiser in optimiser_factory.OptimiserType
    }
    assert actual_optimisers == expected_optimisers


def test_optimiser_params():
    optimiser_params = optimiser_factory.OptimiserParams(
        type=optimiser_factory.OptimiserType.ADAM, params={"learning_rate": 0.01}
    )

    assert optimiser_params.type == optimiser_factory.OptimiserType.ADAM
    assert optimiser_params.params == {"learning_rate": 0.01}

    expected_attributes = {"type", "params"}
    actual_attributes = set(vars(optimiser_params).keys())
    assert actual_attributes == expected_attributes


@pytest.mark.parametrize(
    "data",
    [
        {
            "id": "adam_optimiser",
            "input": optimiser_factory.OptimiserParams(
                type=optimiser_factory.OptimiserType.ADAM,
                params={"learning_rate": 0.01},
            ),
            "expected": keras.optimizers.Adam(learning_rate=0.01),
        },
        {
            "id": "sgd_optimiser",
            "input": optimiser_factory.OptimiserParams(
                type=optimiser_factory.OptimiserType.SGD,
                params={
                    "learning_rate": 0.01,
                    "weight_decay": 0.01 / 10,
                    "momentum": 0.9,
                },
            ),
            "expected": keras.optimizers.SGD(
                learning_rate=0.01, weight_decay=0.01 / 10, momentum=0.9
            ),
        },
        {
            "id": "adagrad_optimiser",
            "input": optimiser_factory.OptimiserParams(
                type=optimiser_factory.OptimiserType.ADAGRAD,
                params={"learning_rate": 0.01},
            ),
            "expected": keras.optimizers.Adagrad(learning_rate=0.01),
        },
    ],
    ids=lambda case: case["id"],
)
class TestOptimiserFactory:

    def test_create_optimiser(self, data):
        observed = optimiser_factory.OptimiserFactory().create_optimiser(data["input"])
        assert isinstance(observed, type(data["expected"]))
