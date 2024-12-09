import keras
import numpy as np
import pandas as pd
import pytest

from bench.models.moma import training, normalisers
from bench.models.moma.preprocessing_utils import integrators


def test_learning_params():
    expected_attributes = {
        "x_train",
        "y_train",
        "x_val",
        "y_val",
        "epochs",
        "batch_size",
        "callbacks",
    }
    learning_params = training.LearningParams(
        x_train=[np.array([1, 2, 3]), np.array([4, 5, 6])],
        y_train=np.array([7, 8, 9]),
        x_val=[np.array([10, 11, 12]), np.array([13, 14, 15])],
        y_val=np.array([16, 17, 18]),
        epochs=10,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping()],
    )
    actual_attributes = set(vars(learning_params).keys())
    assert actual_attributes == expected_attributes


@pytest.fixture
def data():
    return {
        "scaled_train": {
            "feature1": pd.DataFrame([[1, 2], [3, 4]]),
            "feature2": pd.DataFrame([[5, 6], [7, 8]]),
            "target": pd.DataFrame([1, 0]),
        },
        "scaled_test": {
            "feature1": pd.DataFrame([[1, 2], [3, 4]]),
            "feature2": pd.DataFrame([[5, 6], [7, 8]]),
            "target": pd.DataFrame([1, 0]),
        },
        "train": {
            "feature1": pd.DataFrame([[1, 2], [3, 4]]),
            "feature2": pd.DataFrame([[5, 6], [7, 8]]),
            "target": pd.DataFrame([1, 0]),
        },
    }


@pytest.fixture
def x_train():
    result = [
        normalisers.ScaledData(name="proteomics", data=np.array([[1, 2], [3, 4]])),
        normalisers.ScaledData(name="fluxomics", data=np.array([[5, 6], [7, 8]])),
    ]
    return result


@pytest.fixture
def x_val():
    result = [
        normalisers.ScaledData(name="proteomics", data=np.array([[2, 3], [4, 5]])),
        normalisers.ScaledData(name="fluxomics", data=np.array([[6, 7], [8, 9]])),
    ]
    return result


@pytest.fixture
def y_train():
    result = integrators.OmicsData(name="growth", data=pd.DataFrame([1, 0]))
    return result


@pytest.fixture
def y_val():
    result = integrators.OmicsData(name="growth", data=pd.DataFrame([1, 0]))
    return result


def callbacks_equal(callbacks1, callbacks2):
    """
    Compare two lists of Keras callbacks based on their configurations.
    """
    if len(callbacks1) != len(callbacks2):
        return False

    for cb1, cb2 in zip(callbacks1, callbacks2):
        if not (type(cb1) == type(cb2) and cb1.__dict__ == cb2.__dict__):
            return False

    return True


def test_prepare_learning_params(x_train, x_val, y_train, y_val):

    observed = training.prepare_learning_params(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        epochs=10,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping()],
    )

    expected_attributes = {
        "x_train",
        "y_train",
        "x_val",
        "y_val",
        "epochs",
        "batch_size",
        "callbacks",
    }
    observed_attributes = set(vars(observed).keys())
    assert observed_attributes == expected_attributes
    assert all(
        np.array_equal(a.data, b.data) for a, b in zip(observed.x_train, x_train)
    )
    assert all(np.array_equal(a.data, b.data) for a, b in zip(observed.x_val, x_val))
    assert np.array_equal(observed.y_train.data, y_train.data)
    assert np.array_equal(observed.y_val.data, y_val.data)

    assert observed.epochs == 10
    assert observed.batch_size == 32
    assert callbacks_equal(observed.callbacks, [keras.callbacks.EarlyStopping()])
