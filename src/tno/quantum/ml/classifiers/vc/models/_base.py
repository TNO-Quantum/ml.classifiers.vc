"""This module is used to define base class for quantum models.

To add a new model, you should implement a :py:class:`~vc.models.QModel`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from torchtyping import TensorType

    from tno.quantum.utils import BackendConfig


class ModelError(Exception):
    """Module exception."""

    def __init__(self, message: str) -> None:
        """Init ModelError."""
        super().__init__(message)


class QModel(ABC):
    """Abstract base class for quantum models."""

    def __init__(self, backend: BackendConfig, n_classes: int) -> None:
        """Init :py:class:`QModel`.

        Args:
            backend: An instance of :py:class:`~tno.quantum.utils.BackendConfig`, which
              specifies the quantum backend to be used for computations. This includes
              the name of a PennyLane device and optional configuration settings.
            n_classes: The number of target classes for classification.
        """
        self.backend = backend
        self.n_classes = n_classes

    @abstractmethod
    def preprocess(
        self, X: ArrayLike, min_max: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Convert `X` to features. This method should set the `n_qubits` attribute."""

    @abstractmethod
    def get_init_weights(
        self, *, random: bool, random_state: RandomState
    ) -> NDArray[np.float64]:
        """Generate weights to be used as initial trainable parameters."""

    @abstractmethod
    def get_qfunc(self) -> Callable[[TensorType, TensorType], TensorType]:
        """Generate and return a quantum function for the quantum circuit."""
