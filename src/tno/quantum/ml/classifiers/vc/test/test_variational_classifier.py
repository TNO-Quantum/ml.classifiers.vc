"""This module contains tests for the VariationalClassifier class."""

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from tno.quantum.ml.classifiers.vc import VariationalClassifier
from tno.quantum.ml.classifiers.vc.models import ModelError
from tno.quantum.ml.datasets import get_iris_dataset, get_wine_dataset
from tno.quantum.utils import BackendConfig


def test_sklearn_compliance() -> None:
    classifier = VariationalClassifier(random_state=0)
    for estimator, check in check_estimator(classifier, generate_only=True):
        check(estimator)


def _std_scale(
    X_training: NDArray[np.float64], X_validation: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    std_scale = StandardScaler().fit(X_training)
    return std_scale.transform(X_training), std_scale.transform(X_validation)


@pytest.mark.parametrize(
    "model_name", ["expected_value_model", "modulo_model", "parity_model"]
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_variational_classifier_two_classes(
    model_name: str,
    use_bias: bool,
) -> None:
    """Test variational classifier on data with two classes."""
    # Load training data
    X_training, y_training, X_validation, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Preprocess data
    X_training, X_validation = _std_scale(X_training, X_validation)
    X_training = X_training / np.linalg.norm(X_training, ord=2, axis=-1)[:, None]
    X_validation = X_validation / np.linalg.norm(X_validation, ord=2, axis=-1)[:, None]

    # Define classifier
    vc = VariationalClassifier(
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        },
        use_bias=use_bias,
    )

    # Fit the vc
    n_iter = 2
    vc = vc.fit(X_training, y_training, n_iter=n_iter)
    vc.predict(X_training)
    vc.predict(X_validation)
    assert len(vc.history_["loss"]) == n_iter


@pytest.mark.parametrize(
    "model_name", ["expected_value_model", "modulo_model", "parity_model"]
)
def test_variational_classifier_multiple_classes(
    model_name: str,
) -> None:
    """Test variational classifier for data with multiple classes."""
    X_training, y_training, X_validation, _ = get_iris_dataset(
        n_features=4, n_classes=3, random_seed=0
    )
    X_training, X_validation = _std_scale(X_training, X_validation)

    # Define classifier
    vc = VariationalClassifier(
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        }
    )

    # Fit and predict vc
    vc = vc.fit(X_training, y_training, n_iter=1)
    vc.predict(X_training)
    vc.predict(X_validation)


def test_variational_classifier_many_features() -> None:
    """Test variational classifier on data with many features."""
    X_training, y_training, X_validation, _ = get_wine_dataset(
        n_features=13, n_classes=3, random_seed=0
    )
    X_training, X_validation = _std_scale(X_training, X_validation)

    # Define classifier
    vc = VariationalClassifier(
        model={
            "name": "expected_value_model",
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.3},
        },
    )

    # Perform fit and predict
    vc.fit(X_training, y_training, n_iter=1)
    vc.predict(X_training)
    vc.predict(X_validation)


def test_warm_fit() -> None:
    """Test variational classifier using warm init."""
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )

    # Define classifier
    vc = VariationalClassifier(warm_init=True)

    # Perform fit and predict
    for _ in range(2):
        vc.fit(X_training, y_training, n_iter=1)
        vc.predict(X_training)

    assert len(vc.history_["loss"]) == 2


def test_cold_fit() -> None:
    """Test variational classifier using cold init."""
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )
    vc = VariationalClassifier(warm_init=False)

    # Perform fit and predict
    for _ in range(2):
        vc.fit(X_training, y_training, n_iter=1)
        vc.predict(X_training)

    assert len(vc.history_["loss"]) == 1


@pytest.mark.parametrize(
    "model_name", ["modulo_model", "expected_value_model", "parity_model"]
)
def test_non_random_init(model_name: str) -> None:
    """Test the random init is False parameter."""
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )
    vc = VariationalClassifier(
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        },
        random_init=False,
    )
    vc = vc.fit(X_training, y_training, n_iter=1)


def test_non_default_backend() -> None:
    """Test a qiskit backend."""
    if "qiskit.aer" in BackendConfig.supported_items():
        X_training, y_training, _, _ = get_iris_dataset(
            n_features=2, n_classes=2, random_seed=0
        )
        backend = {"name": "qiskit.aer", "options": {}}
        vc = VariationalClassifier(backend=backend)
        vc.fit(X_training, y_training, n_iter=1)


# region warnings and errors


def test_one_class_warning() -> None:
    """Test raise warning if data has only one class."""
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=1, random_seed=0
    )

    vc = VariationalClassifier()
    expected_message = "one class found"
    with pytest.warns(UserWarning, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


@pytest.mark.parametrize(
    "model_name", ["modulo_model", "expected_value_model", "parity_model"]
)
def test_maximum_number_of_classes(model_name: str) -> None:
    """Test raise error if number of classes is to large."""
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=1, n_classes=3, random_seed=0
    )

    vc = VariationalClassifier(
        model={
            "name": model_name,
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        }
    )
    expected_message = "number of classes should be less than or equal to"
    with pytest.raises(ModelError, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


def test_cold_warm_fit() -> None:
    """Test raise warning vc is not yet fitted with warm init is True."""
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )
    vc = VariationalClassifier(warm_init=True)

    expected_message = "fit method has not been called"
    with pytest.warns(UserWarning, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


def test_invalid_optimizer() -> None:
    """Test provide invalid optimizer."""
    X_training, y_training, _, _ = get_iris_dataset(
        n_features=2, n_classes=2, random_seed=0
    )
    # Define classifier with invalid optimizer
    vc = VariationalClassifier(optimizer={"name": "eve", "options": {}})

    expected_message = "Name 'eve' does not match any of the supported items."
    with pytest.raises(KeyError, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)


def test_invalid_model() -> None:
    """Test provide invalid model."""
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

    expected_message = "Name 'bad_model' does not match any of the supported items."
    with pytest.raises(KeyError, match=expected_message):
        vc = vc.fit(X_training, y_training, n_iter=1)
