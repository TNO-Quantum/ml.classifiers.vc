"""This module contains tests for the VariationalClassifier class."""
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from tno.quantum.ml.classifiers.vc import VariationalClassifier
from tno.quantum.ml.classifiers.vc.models import ModelError, ParityModel
from tno.quantum.ml.classifiers.vc.utils import get_bin
from tno.quantum.ml.datasets import get_iris_dataset, get_wine_dataset

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-arguments


def test_sklearn_compliance() -> None:
    classifier = VariationalClassifier(random_state=0)
    for estimator, check in check_estimator(classifier, generate_only=True):
        print(check)
        check(estimator)


def _accuracy(labels: NDArray[Any], predictions: NDArray[Any]) -> np.float_:
    return np.sum(np.isclose(labels, predictions)) / labels.size


def _std_scale(
    X_training: NDArray[np.float_], X_validation: NDArray[np.float_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    std_scale = StandardScaler().fit(X_training)
    return std_scale.transform(X_training), std_scale.transform(X_validation)


@pytest.mark.parametrize(
    "backend_fit,backend_predict,model_name,use_bias,n_iterations",
    [
        (
            {"name": "default.qubit", "options": {}},
            {"name": "default.qubit", "options": {}},
            "expected_values_model",
            True,
            [150],
        ),
        (
            {"name": "default.qubit", "options": {}},
            {"name": "default.qubit", "options": {}},
            "expected_values_model",
            False,
            [160],
        ),
        (
            {"name": "default.qubit", "options": {}},
            {"name": "default.qubit", "options": {}},
            "probabilities_model",
            False,
            [160],
        ),
        (
            {"name": "default.qubit", "options": {}},
            {"name": "default.qubit", "options": {}},
            "parity_model",
            False,
            [160],
        ),
        (
            {"name": "default.qubit", "options": {}},
            {"name": "qiskit.aer", "options": {}},
            "expected_values_model",
            False,
            [200],
        ),
    ],
)
def test_variational_classifier_two_classes(
    backend_fit: Dict[str, Any],
    backend_predict: Dict[str, Any],
    model_name: str,
    use_bias: bool,
    n_iterations: List[int],
) -> None:
    # Load training data
    X_training, y_training, X_validation, y_validation = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Preprocess data
    X_training, X_validation = _std_scale(X_training, X_validation)
    X_training = X_training / np.linalg.norm(X_training, ord=2, axis=-1)[:, None]
    X_validation = X_validation / np.linalg.norm(X_validation, ord=2, axis=-1)[:, None]

    # Define classifier
    vc = VariationalClassifier(
        batch_size=5,
        backend=backend_fit,
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        },
        optimizer={"name": "adam", "options": {}},
        use_bias=use_bias,
        random_init=True,
        warm_init=True,
        random_state=2,
    )

    # Fit and re-fit by making use of the parameters fitted in the previous iteration
    for n_iter in n_iterations:
        vc = vc.fit(X_training, y_training, n_iter=n_iter)

        # Predict
        vc.set_params(backend=backend_predict)
        predictions_training = vc.predict(X_training)
        predictions_validation = vc.predict(X_validation)
        vc.set_params(backend=backend_fit)

        # Compute accuracy
        acc_training = _accuracy(y_training, predictions_training)
        acc_validation = _accuracy(y_validation, predictions_validation)
        print(
            f"Acc training: {acc_training:0.7f} | Acc validation: {acc_validation:0.7f}"
        )

    assert acc_training >= 0.8
    assert acc_validation >= 0.8


@pytest.mark.parametrize(
    "backend,model_name,n_iterations,min_accuracy",
    [
        (
            {"name": "default.qubit", "options": {}},
            "expected_values_model",
            [80],
            0.9,
        ),
        (
            {"name": "default.qubit", "options": {}},
            "probabilities_model",
            [80],
            0.8,
        ),
        (
            {"name": "default.qubit", "options": {}},
            "parity_model",
            [80],
            0.8,
        ),
    ],
)
def test_variational_classifier_multiple_classes(
    backend: Dict[str, Any],
    model_name: str,
    n_iterations: List[int],
    min_accuracy: float,
) -> None:
    # Load training data
    X_training, y_training, X_validation, y_validation = get_iris_dataset(
        n_features=4, n_classes=3, random_seed=0
    )

    # Preprocess
    X_training, X_validation = _std_scale(X_training, X_validation)

    # Define classifier
    vc = VariationalClassifier(
        batch_size=5,
        backend=backend,
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        },
        optimizer={
            "name": "stochastic_gradient_descent",
            "options": {"lr": 0.01, "momentum": 0.9},
        },
        random_init=True,
        warm_init=True,
        random_state=0,
    )

    # Fit and re-fit by making use of the parameters fitted in the previous iteration
    for n_iter in n_iterations:
        vc = vc.fit(X_training, y_training, n_iter=n_iter)

        # Predict
        predictions_training = vc.predict(X_training)
        predictions_validation = vc.predict(X_validation)

        # Compute accuracy
        acc_training = _accuracy(y_training, predictions_training)
        acc_validation = _accuracy(y_validation, predictions_validation)
        print(
            f"Acc training: {acc_training:0.7f} | Acc validation: {acc_validation:0.7f}"
        )

    assert acc_training > min_accuracy and acc_validation > min_accuracy


@pytest.mark.parametrize(
    "backend,n_iterations,min_accuracy",
    [
        (
            {"name": "default.qubit", "options": {}},
            [80],
            0.7,
        ),
    ],
)
def test_variational_classifier_many_features(
    backend: Dict[str, Any], n_iterations: List[int], min_accuracy: float
) -> None:
    # Load training data
    X_training, y_training, X_validation, y_validation = get_wine_dataset(
        n_features=13, n_classes=3, random_seed=0
    )

    # Preprocess
    X_training, X_validation = _std_scale(X_training, X_validation)

    # Define classifier
    vc = VariationalClassifier(
        batch_size=10,
        backend=backend,
        model={
            "name": "expected_values_model",
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.3},
        },
        optimizer={"name": "adam", "options": {}},
        random_init=True,
        warm_init=True,
        random_state=0,
    )

    # Fit and re-fit by making use of the parameters fitted in the previous iteration
    for n_iter in n_iterations:
        vc = vc.fit(X_training, y_training, n_iter=n_iter)

        # Predict
        predictions_training = vc.predict(X_training)
        predictions_validation = vc.predict(X_validation)

        # Compute accuracy
        acc_training = _accuracy(y_training, predictions_training)
        acc_validation = _accuracy(y_validation, predictions_validation)
        print(
            f"Acc training: {acc_training:0.7f} | Acc validation: {acc_validation:0.7f}"
        )

    assert acc_training > min_accuracy and acc_validation > min_accuracy


def test_one_class_warning() -> None:
    # Load training data
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=1, random_seed=0
    )

    # Define classifier
    vc = VariationalClassifier()

    # Fit
    expected_message = "one class found"
    with pytest.warns(UserWarning, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


@pytest.mark.parametrize(
    "model_name", ["probabilities_model", "expected_values_model", "parity_model"]
)
def test_maximum_number_of_classes(model_name: str) -> None:
    # Load training data
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=1, n_classes=3, random_seed=0
    )

    # Define classifier
    vc = VariationalClassifier(
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        }
    )

    # Fit
    expected_message = "number of classes should be less than or equal to"
    with pytest.raises(ModelError, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


def test_warm_fit() -> None:
    # Load training data
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Define classifier
    vc = VariationalClassifier(warm_init=True)

    # Fit
    for _ in range(2):
        vc = vc.fit(X_training, y_training, n_iter=1)

    assert True


def test_cold_warm_fit() -> None:
    # Load training data
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Define classifier
    vc = VariationalClassifier(warm_init=True)

    # Fit
    expected_message = "fit method has not been called"
    with pytest.warns(UserWarning, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


@pytest.mark.parametrize("model_name", ["probabilities_model", "expected_values_model"])
def test_non_random_init(model_name: str) -> None:
    # Load training data
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Define classifier
    vc = VariationalClassifier(
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        },
        random_init=False,
    )

    # Fit
    vc = vc.fit(X_training, y_training, n_iter=1)

    assert True


@pytest.mark.parametrize("num_qubits", [1, 2, 5, 10])
def test_class_assignment_parity_model_two_classes(num_qubits: int) -> None:
    for i in range(2**num_qubits):
        bit_array = get_bin(i, num_qubits)
        model_assignment = ParityModel.get_class_from_bit_array(bit_array, m=1)
        parity = sum(int(_) for _ in bit_array) % 2
        assert parity == model_assignment


@pytest.mark.parametrize(
    "assignment_dict,m",
    [
        (
            {
                0: ["000", "110"],
                1: ["010", "100"],
                2: ["001", "111"],
                3: ["011", "101"],
            },
            2,
        ),
        (
            {
                0: ["000"],
                1: ["100"],
                2: ["010"],
                3: ["110"],
                4: ["001"],
                5: ["101"],
                6: ["011"],
                7: ["111"],
            },
            3,
        ),
        (
            {
                0: ["0000", "0110", "1010", "1100"],
                1: ["0010", "0100", "1000", "1110"],
                2: ["0001", "0111", "1011", "1101"],
                3: ["0011", "0101", "1001", "1111"],
            },
            2,
        ),
        (
            {
                0: ["00000", "01100", "10100", "11000"],
                1: ["00100", "01000", "10000", "11100"],
                2: ["00010", "01110", "10110", "11010"],
                3: ["00110", "01010", "10010", "11110"],
                4: ["00001", "01101", "10101", "11001"],
                5: ["00101", "01001", "10001", "11101"],
                6: ["00011", "01111", "10111", "11011"],
                7: ["00111", "01011", "10011", "11111"],
            },
            3,
        ),
    ],
)
def test_class_assignment_parity_model(
    assignment_dict: Dict[int, List[str]], m: int
) -> None:
    for class_id, bit_strings in assignment_dict.items():
        for bit_string in bit_strings:
            bit_array = np.array([int(_) for _ in bit_string])
            model_assignment = ParityModel.get_class_from_bit_array(bit_array, m)
            assert model_assignment == class_id


def test_invalid_optimizer() -> None:
    # Load training data
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Define classifier with invalid optimizer
    vc = VariationalClassifier(optimizer={"name": "eve", "options": {}})

    # Fit
    expected_message = "Invalid optimizer name."
    with pytest.raises(ValueError, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


def test_invalid_model() -> None:
    # Load training data
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Define classifier with invalid model
    vc = VariationalClassifier(
        model={
            "name": "bad_model",
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        }
    )

    # Fit
    expected_message = "Invalid model name."
    with pytest.raises(ValueError, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)
