"""This module implements a scikit-learn compatible, variational quantum classifier."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn
from tqdm import trange

from tno.quantum.ml.classifiers.vc.models import QModel
from tno.quantum.utils import (
    BackendConfig,
    BaseConfig,
    OptimizerConfig,
    get_installed_subclasses,
)
from tno.quantum.utils.validation import check_random_state

if TYPE_CHECKING:
    from sklearn.utils._tags import Tags


def get_default_backend_if_none(
    backend: BackendConfig | Mapping[str, Any] | None = None,
) -> BackendConfig:
    """Set default training backend if the one provided is ``None``.

    Default training backend ``{"name": "default.qubit", "options": {}}``.

    Args:
        backend: backend configuration or ``None``.

    Raises:
        KeyError: If `backend` does not contain key ``"name"``.

    Returns:
        Given backend or the default training backend.
    """
    return BackendConfig.from_mapping(
        backend if backend is not None else {"name": "default.qubit"}
    )


def get_default_optimizer_if_none(
    optimizer: OptimizerConfig | Mapping[str, Any] | None = None,
) -> OptimizerConfig:
    """Set default optimizer if the one provided is ``None``.

    Default optimizer ``{"name": "adagrad", "options": {}}``.

    Args:
        optimizer: optimizer configuration or ``None``.

    Raises:
        KeyError: If `optimizer` does not contain key ``"name"``.

    Returns:
        Given optimizer or the default optimizer.
    """
    return OptimizerConfig.from_mapping(
        optimizer if optimizer is not None else {"name": "adagrad"}
    )


def get_default_model_if_none(
    model: ModelConfig | Mapping[str, Any] | None = None,
) -> ModelConfig:
    """Set default model if the one provided is ``None``.

    Default model ``{"name": "modulo_model", "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5}}``.

    Args:
        model: model configuration or ``None``.

    Raises:
        KeyError: If `model` does not contain key ``"name"``.

    Returns:
        Given model or the default model.
    """  # noqa: E501
    default_model = {
        "name": "modulo_model",
        "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
    }
    return ModelConfig.from_mapping(model if model is not None else default_model)


@dataclass(init=False)
class ModelConfig(BaseConfig[QModel]):
    """Model configuration for creating a model.

    Supported models can be found by calling
    :py:meth:`~ModelConfig.supported_items`.

    Attributes:
        name: is used to determine the
            :py:class:`~tno.quantum.ml.classifiers.vc.models.QModel` class.
        options: keyword arguments to be passed to the constructor of the
            :py:class:`~tno.quantum.ml.classifiers.vc.models.QModel` class.

    Example:
        >>> backend_config = BackendConfig(name="default.qubit", options={"wires": 1})
        >>> model_config = ModelConfig(name="modulo_model")
        >>> model_instance = model_config.get_instance(backend=backend_config, n_classes=2)
    """  # noqa:E501

    def __init__(self, name: str, options: Mapping[str, Any] | None = None) -> None:
        """Init :py:class:`ModelConfig`.

        Args:
            name: The name of the quantum model to be used.
            options: A dict containing configuration options for the quantum model.

        Raises:
            TypeError: If `name` is not a string.
            TypeError: If `options` is not a mapping.
            KeyError: If `options` contains a key that is not a string.
            ValueError: If `name` does not adhere to the snake_case convention.
        """
        super().__init__(name=name, options=options)

    @staticmethod
    def supported_items() -> dict[str, Callable[..., QModel]]:
        """Obtain all supported QModels.

        Returns:
            Dict with all supported quantum models.
        """
        supported_optimizers: dict[str, Callable[..., QModel]] = {}

        for name, cls in get_installed_subclasses(
            "tno.quantum.ml.classifiers.vc.models.expected_value_model", QModel
        ).items():
            if issubclass(cls, QModel):
                supported_optimizers[name] = cls  # noqa: PERF403

        for name, cls in get_installed_subclasses(
            "tno.quantum.ml.classifiers.vc.models.probability_model", QModel
        ).items():
            if issubclass(cls, QModel):
                supported_optimizers[name] = cls  # noqa: PERF403

        return supported_optimizers


class VariationalClassifier(ClassifierMixin, BaseEstimator):  # type:ignore[misc]
    """Variational classifier."""

    def __init__(  # noqa: PLR0913
        self,
        batch_size: int = 5,
        backend: BackendConfig | Mapping[str, Any] | None = None,
        model: ModelConfig | Mapping[str, Any] | None = None,
        optimizer: OptimizerConfig | Mapping[str, Any] | None = None,
        *,
        use_bias: bool = False,
        random_init: bool = True,
        warm_init: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Init :py:class:`VariationalClassifier`.

        Args:
            batch_size: batch size to be used during fitting.
            backend: The backend to perform the model on, see docstring of
                :py:class:`~tno.quantum.utils.BackendConfig`. Defaults to
                ``{"name": "default.qubit", "options": {}}``.
            model: The quantum model, see docstring of
                :py:class:`~tno.quantum.ml.classifiers.vc.ModelConfig`.
                Defaults to ``{"name": "modulo_model", "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5}}``.
            optimizer: The classical optimizer, see docstring of
                :py:class:`~tno.quantum.utils.OptimizerConfig`. Defaults to
                ``{"name": "adagrad", "options": {}}``.
            use_bias: set to ``True`` if a bias parameter should be optimized over.
            random_init: set to ``True`` if parameters to optimize over should be
              initialized randomly.
            warm_init: set to ``True`` if parameters from a previous call to fit
              should be used.
            random_state: random seed for repeatability.
        """  # noqa: E501
        self.batch_size = batch_size
        self.backend = backend
        self.model = model
        self.optimizer = optimizer
        self.use_bias = use_bias
        self.random_init = random_init
        self.warm_init = warm_init
        self.random_state = random_state

        self.history_: dict[str, list[float]]

    def fit(  # noqa: PLR0915
        self,
        X: ArrayLike,
        y: ArrayLike,
        n_iter: int = 1,
        *,
        verbose: bool = True,
    ) -> VariationalClassifier:
        """Fit data using a quantum model.

        Args:
            X: training data with shape (`n_samples`, `n_features`).
            y: target values with shape (`n_samples`,).
            n_iter: number of training iterations.
            verbose: whether to display a progress bar during training.
        """
        # Check
        X, y = validate_data(self, X, y)
        X, y = np.array(X), np.array(y)
        if X.shape[0] == 1:
            error_msg = "Cannot fit with only 1 sample."
            raise ValueError(error_msg)

        # Get default settings if necessary
        model = get_default_model_if_none(self.model)
        backend = get_default_backend_if_none(self.backend)
        optimizer = get_default_optimizer_if_none(self.optimizer)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if self.classes_.size == 1:
            warnings.warn("Only one class found. Fitting anyway.", stacklevel=2)

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.classes_)
        y = self.label_encoder_.transform(y)

        # Get numpy's random state
        random_state = check_random_state(self.random_state, name="random_state")

        # Get quantum model
        qmodel = model.get_instance(backend=backend, n_classes=self.classes_.size)

        # Preprocess X and y
        self.min_max_ = (np.min(X, axis=0), np.max(X, axis=0))
        X_preprocessed = torch.tensor(
            qmodel.preprocess(X, self.min_max_), requires_grad=False
        )
        y = torch.tensor(y, requires_grad=False)

        # Train
        weights_init = qmodel.get_init_weights(
            random=self.random_init, random_state=random_state
        )
        bias_init = np.zeros(1 if self.classes_.size == 2 else (self.classes_.size,))
        optimizer_state = None
        if self.warm_init:
            try:
                check_is_fitted(self, ["weights_", "bias_"])
            except NotFittedError:
                warnings.warn(
                    "The warm_init keyword is set to True, but the fit"
                    " method has not been called before. Fitting will be"
                    " performed for the first time.",
                    stacklevel=2,
                )
                self._reset_history()
            else:
                weights_init = self.weights_
                bias_init = self.bias_
                optimizer_state = self.optimizer_state_
        else:
            self._reset_history()

        weights = torch.tensor(weights_init, requires_grad=True)
        bias = torch.tensor(bias_init, requires_grad=self.use_bias)

        qfunc = qmodel.get_qfunc()
        opt_params = [weights]
        if self.use_bias:
            opt_params.append(bias)
        opt = optimizer.get_instance(params=opt_params)
        if optimizer_state is not None:
            opt.load_state_dict(optimizer_state)

        training_loop = (
            trange(n_iter, desc="Training iteration: ") if verbose else range(n_iter)
        )

        for _ in training_loop:
            # Load batch
            batch_index = random_state.permutation(range(y.numel()))[: self.batch_size]
            X_preprocessed_batch = X_preprocessed[batch_index, :]
            y_batch = y[batch_index]

            # Compute predictions and loss
            predictions = torch.stack(
                [
                    (prediction[0] if isinstance(prediction, tuple) else prediction)
                    + bias
                    for prediction in (qfunc(weights, x) for x in X_preprocessed_batch)
                ]
            ).squeeze()

            loss_func: torch.nn.modules.loss._WeightedLoss
            if self.classes_.size > 2:
                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(predictions, y_batch.long())
            else:
                loss_func = nn.BCELoss()
                loss = loss_func(nn.Sigmoid()(predictions), y_batch.double())

            self.history_["loss"].append(float(loss.detach()))

            # Backpropagation step
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Store fit parameters
        self.weights_: NDArray[np.float64] = weights.detach().numpy()
        self.bias_: NDArray[np.float64] = bias.detach().numpy()
        self.optimizer_state_: dict[str, Any] = opt.state_dict()
        self.n_features_in_: int = X.shape[1]

        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class using a quantum model.

        Args:
            X: input data with shape (`n_samples`  `n_features`).

        Returns:
            The predicted classes with shape (`n_samples`, `n_classes`).
        """
        # Validate
        X = np.asarray(validate_data(self, X, reset=False))

        # Check
        check_is_fitted(self, ["weights_", "bias_"])

        # Get default settings if necessary
        model = get_default_model_if_none(self.model)
        backend = get_default_backend_if_none(self.backend)

        # Get quantum model
        qmodel = model.get_instance(backend=backend, n_classes=self.classes_.size)

        # Preprocess X
        X_preprocessed = torch.tensor(
            qmodel.preprocess(X, self.min_max_), requires_grad=False
        )

        # Load weights and bias
        weights = torch.tensor(self.weights_, requires_grad=False)
        bias = torch.tensor(self.bias_, requires_grad=False)

        # Predict
        qfunc = qmodel.get_qfunc()
        predictions_tensor = torch.stack(
            [
                (prediction[0] if isinstance(prediction, tuple) else prediction) + bias
                for prediction in (qfunc(weights, x) for x in X_preprocessed)
            ]
        ).squeeze()

        if self.classes_.size > 2:
            predictions_tensor_2d = torch.atleast_2d(predictions_tensor)  # type: ignore[no-untyped-call]
            predicted_idx = torch.max(torch.softmax(predictions_tensor_2d, dim=1), 1)[
                1
            ].numpy()
        else:
            predicted_idx = (torch.sigmoid(predictions_tensor) > 1 / 2).int().numpy()

        predictions: NDArray[Any]
        if self.classes_.size == 1:
            predictions = np.full(shape=X.shape[0], fill_value=self.classes_[0])
        else:
            predictions = self.label_encoder_.inverse_transform(predicted_idx)

        return predictions

    def _reset_history(self) -> None:
        """Reset the history."""
        self.history_ = {"loss": []}

    def __sklearn_tags__(self) -> Tags:
        """Return estimator tags for use in sklearn tests."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags
