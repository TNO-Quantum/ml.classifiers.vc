"""This module implements a scikit-learn compatible, variational quantum classifier.

Usage:

.. code-block::

    vc = VariationalClassifier()
    vc = vc.fit(X_training, y_training, n_iter=60)
    predictions = vc.predict(X_validation)
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import nn
from tqdm import tqdm

from tno.quantum.ml.classifiers.vc import models as quantum_models
from tno.quantum.ml.classifiers.vc import optimizers as quantum_optimizers

# pylint: disable=attribute-defined-outside-init
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals


def get_default_if_none(
    backend: Optional[Dict[str, Any]] = None,
    model: Optional[Dict[str, Any]] = None,
    optimizer: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Set default value if the one provided is ``None``.

    Args:
        backend: see docstring of :py:func:`~vc.models.get_model`.
          default value ``{"name": "default.qubit", "options": {}}``.
        model: see docstring of :py:func:`~vc.models.get_model`
          default value ``{"name": "modulo_model",
          "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5}}``.
        optimizer: see docstring of :py:func:`~vc.optimizers.get_optimizer`
          default value ``{"name": "adam", "options": {}}``.

    Returns:
        Default backend, model, optimizer.
    """
    if backend is None:
        backend = {"name": "default.qubit", "options": {}}
    if model is None:
        model = {
            "name": "modulo_model",
            "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
        }
    if optimizer is None:
        optimizer = {"name": "adam", "options": {}}
    return backend, model, optimizer


class VariationalClassifier(ClassifierMixin, BaseEstimator):  # type:ignore[misc]
    """Variational classifier."""

    def __init__(
        self,
        batch_size: int = 5,
        backend: Optional[Dict[str, Any]] = None,
        model: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Dict[str, Any]] = None,
        use_bias: bool = False,
        random_init: bool = True,
        warm_init: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Init VariationalClassifier.

        The default values for ``backend``, ``model``, and ``optimizer`` are
        defined in :py:func:`vc.variational_classifier.get_default_if_none`.

        Args:
            batch_size: batch size to be used during fitting.
            backend: see docstring of :py:func:`~vc.models.get_model`.
            model: see docstring of :py:func:`~vc.models.get_model`.
            optimizer: see docstring of :py:func:`~vc.optimizers.get_optimizer`
            use_bias: set to ``True`` if a bias parameter should be optimized over.
            random_init: set to ``True`` if parameters to optimize over should be
              initialized randomly.
            warm_init: set to ``True`` if parameters from a previous call to fit
              should be used.
            random_state: random seed for repeatability.
        """
        self.batch_size = batch_size
        self.backend = backend
        self.model = model
        self.optimizer = optimizer
        self.use_bias = use_bias
        self.random_init = random_init
        self.warm_init = warm_init
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike, n_iter: int = 1) -> VariationalClassifier:
        """Fit data using a quantum model.

        Args:
            X: training data with shape (`n_samples`, `n_features`).
            y: target values with shape (`n_samples`,).
            n_iter: number of training iterations.
        """
        # Check
        X, y = check_X_y(X, y)
        X, y = np.array(X), np.array(y)
        if X.shape[0] == 1:
            raise ValueError("Cannot fit with only 1 sample.")

        # Get default settings if necessary
        backend, model, optimizer = get_default_if_none(
            self.backend, self.model, self.optimizer
        )

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if self.classes_.size == 1:
            warnings.warn("Only one class found. Fitting anyway.")

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(self.classes_)
        y = self.label_encoder_.transform(y)

        # Get numpy's random state
        random_state = check_random_state(self.random_state)

        # Get quantum model
        qmodel = quantum_models.get_model(model, backend, self.classes_.size)

        # Preprocess X and y
        self.min_max_ = (np.min(X, axis=0), np.max(X, axis=0))
        X_preprocessed = torch.tensor(
            qmodel.preprocess(X, self.min_max_), requires_grad=False
        )
        y = torch.tensor(y, requires_grad=False)

        # Train
        weights_init = qmodel.get_init_weights(self.random_init, random_state)
        bias_init = np.zeros(1 if self.classes_.size == 2 else (self.classes_.size,))
        optimizer_state = None
        if self.warm_init:
            try:
                check_is_fitted(self, ["weights_", "bias_"])
            except NotFittedError:
                warnings.warn(
                    "The warm_init keyword is set to True, but the fit"
                    " method has not been called before. Fitting will be"
                    " performed for the first time."
                )
            else:
                weights_init = self.weights_  # pylint: disable=E0203
                bias_init = self.bias_  # pylint: disable=E0203
                optimizer_state = self.optimizer_state_  # pylint: disable=E0203
        weights = torch.tensor(weights_init, requires_grad=True)
        bias = torch.tensor(bias_init, requires_grad=self.use_bias)

        qfunc = qmodel.get_qfunc()
        opt_params = [weights]
        if self.use_bias:
            opt_params.append(bias)
        opt = quantum_optimizers.get_optimizer(opt_params, optimizer)
        if optimizer_state is not None:
            opt.load_state_dict(optimizer_state)

        for _ in tqdm(range(n_iter), desc="Training iteration: "):
            # Load batch
            batch_index = random_state.permutation(range(y.numel()))[: self.batch_size]
            X_preprocessed_batch = X_preprocessed[batch_index, :]
            y_batch = y[batch_index]

            # Compute predictions and loss
            predictions = torch.stack(
                [qfunc(weights, x) + bias for x in X_preprocessed_batch]
            ).squeeze()

            loss_func: torch.nn.modules.loss._WeightedLoss
            if self.classes_.size > 2:
                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(predictions, y_batch.long())
            else:
                loss_func = nn.BCELoss()
                loss = loss_func(nn.Sigmoid()(predictions), y_batch.double())

            # Backpropagation step
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Store fit parameters
        self.weights_: NDArray[np.float_] = weights.detach().numpy()
        self.bias_: NDArray[np.float_] = bias.detach().numpy()
        self.optimizer_state_: Dict[str, Any] = opt.state_dict()
        self.n_features_in_: int = X.shape[1]

        # Return the classifier
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class using a quantum model.

        Args:
            X: input data with shape (`n_samples`  `n_features`).

        Returns:
            The predicted classes with shape (`n_samples`, `n_classes`).
        """
        # Check
        check_is_fitted(self, ["weights_", "bias_"])
        X = np.array(check_array(X))

        # Get default settings if necessary
        backend, model, _ = get_default_if_none(
            self.backend, self.model, self.optimizer
        )

        # Get quantum model
        qmodel = quantum_models.get_model(model, backend, self.classes_.size)

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
            [qfunc(weights, x) + bias for x in X_preprocessed]
        ).squeeze()

        if self.classes_.size > 2:
            predictions_tensor_2d = torch.atleast_2d(
                predictions_tensor
            )  # type: ignore[no-untyped-call]
            predicted_idx = torch.max(torch.softmax(predictions_tensor_2d, dim=1), 1)[
                1
            ].numpy()
        else:
            predicted_idx = (torch.sigmoid(predictions_tensor) > 0.5).int().numpy()

        predictions: NDArray[Any]
        if self.classes_.size == 1:
            predictions = np.full(shape=X.shape[0], fill_value=self.classes_[0])
        else:
            predictions = self.label_encoder_.inverse_transform(predicted_idx)

        return predictions

    def _more_tags(self) -> Dict[str, bool]:
        """Return estimator tags for use in sklearn tests."""
        return {
            "poor_score": True,  # We have our own tests
        }
