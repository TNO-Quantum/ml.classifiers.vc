"""This module contains quantum models that are based on computing expected values for class assignment."""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pennylane
import pennylane.measurements
import torch
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray

from tno.quantum.ml.classifiers.vc.models import ModelError, QModel

if TYPE_CHECKING:
    from torchtyping import TensorType

    from tno.quantum.utils import BackendConfig


class ExpectedValueModel(QModel):
    r"""Model using angle encoding and expected values.

    This model implements a quantum unitary transformation U(x) composed of trainable
    and encoding layers, where the expected values of measured qubits are used as output
    features.

    The model is structured as:

    .. math::
        U(x) = W_{L + 1} \cdot S(x) \cdot W_{L} \cdot \ldots
        \cdot W_{2} \cdot S(x) \cdot W_{1}

    where:
        - $x$ represents input data,
        - $S(x)$ is an encoding circuit that maps classical data to quantum states,
        - $W$ represents trainable unitary transformations applied to the qubits.

    Attributes:
        backend: Configuration for the quantum computation backend.
        n_classes: Number of output classes.
        n_layers: Number of layers in the unitary circuit.
        n_trainable_sublayers: Number of trainable layers in each W block.
        scaling: Scaling factor for preprocessing input data.
    """

    def __init__(
        self,
        backend: BackendConfig,
        n_classes: int,
        n_layers: int = 2,
        n_trainable_sublayers: int = 2,
        scaling: float = 0.5,
    ) -> None:
        r"""Init :py:class:`ExpectedValueModel`.

        This model implements a unitary $U(x)$ of the form:

        .. math::
            U(x) = W_{L + 1} \cdot S(x) \cdot W_{L} \cdot \ldots
            \cdot W_{2} \cdot S(x) \cdot W_{1}

        where:

            - $x$ is the data to encode,
            - $S$ is an encoding circuit,
            - $W$ is a trainable circuit.

        Args:
            backend: A backend configuration, which specifies the quantum backend to be
                used for computations. This includes the name of a PennyLane device and
                optional configuration settings.
            n_classes: The number of target classes for classification.
            n_layers: number of layers in $U(x)$ (equal to $L$).
            n_trainable_sublayers: number of layers for each $W$.
            scaling: scaling to apply to the data, before applying angle embedding.
        """
        super().__init__(backend, n_classes)
        self.n_layers = n_layers
        self.n_trainable_sublayers = n_trainable_sublayers
        self.scaling = scaling

    def preprocess(
        self, X: ArrayLike, min_max: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        r"""Maps input `X` in the range `min_max` to $(-\pi, \pi]$.

        Args:
            X: input data with shape (`n_samples`, `n_features`).
            min_max: minimum value and maximum value.
        """
        # Coerce into an ndarray
        X = np.asarray(X)

        # Convert to angles between -pi and pi
        angles = 2 * np.pi * (X - min_max[0]) / (min_max[1] - min_max[0]) - np.pi

        # Set number of qubits required by model and validate
        self.n_qubits = angles.shape[1]
        if self.n_classes > self.n_qubits:
            error_msg = (
                "The number of classes should be less than or equal to the number "
                "of features."
            )
            raise ModelError(error_msg)

        return angles

    def _S(self, x: TensorType) -> None:
        """Define encoding circuit."""
        pennylane.AngleEmbedding(
            features=self.scaling * x, wires=range(self.n_qubits), rotation="X"
        )

    def _W(self, w: TensorType) -> None:
        """Define trainable circuit."""
        pennylane.StronglyEntanglingLayers(w, wires=range(self.n_qubits))

    def get_init_weights(
        self, *, random: bool, random_state: RandomState
    ) -> NDArray[np.float64]:
        r"""Get init weights between $0$ and $2\pi$."""
        array_shape = self.n_layers + 1, self.n_trainable_sublayers, self.n_qubits, 3
        if random:
            return 2 * np.pi * random_state.random(size=array_shape)

        return np.zeros(array_shape)

    def _circuit(
        self,
        weights: TensorType,
        x: TensorType,
    ) -> tuple[pennylane.measurements.MeasurementProcess, ...]:
        """Create modelled circuit.

        Args:
            weights: weights for trainable circuits
            x: data for the encoding circuit

        Returns:
            Measurements (expected values).
        """
        # Define circuit
        for w in weights[:-1, :, :, :]:
            self._W(w)
            self._S(x)
        self._W(weights[-1, :, :, :])

        # Measure the expected value for as many qubits as classes we have
        wires = range(self.n_classes) if self.n_classes > 2 else range(1)

        return tuple(pennylane.expval(pennylane.PauliZ(wire)) for wire in wires)

    def get_qfunc(self) -> Callable[[TensorType, TensorType], TensorType]:  # noqa: D102
        dev = self.backend.get_instance(wires=self.n_qubits)
        qnode = pennylane.QNode(self._circuit, dev, interface="torch")

        def _process_measurement(
            weights: TensorType, x: TensorType, qnode: pennylane.QNode = qnode
        ) -> TensorType:
            if self.n_classes > 2:
                return torch.stack(qnode(weights, x))
            return qnode(weights, x)

        return _process_measurement
