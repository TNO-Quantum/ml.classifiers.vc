"""This module contains quantum models that are based on probability measurements.

Two different post-processing strategies are defined:

- :py:class:`~tno.quantum.ml.classifiers.vc.models.probability_model.ModuloModel`:
  Class assignment of a bit string based on its decimal representation. If needed, the
  decimal representation is reduced modulo the number of classes.
- :py:class:`~tno.quantum.ml.classifiers.vc.models.probability_model.ParityModel`:
  Class assignment of a bit string based on the decimal representation of its bits. If
  more than log number of classes qubits are measured, the number of bits are combined
  by considering its parity.

.. list-table:: Class Assignments (3 bits, 4 classes)
   :header-rows: 1

   * - Bit String
     - ModuloModel Class
     - ParityModel Class
   * - 000
     - 0
     - 0
   * - 001
     - 1
     - 2
   * - 010
     - 2
     - 1
   * - 011
     - 3
     - 3
   * - 100
     - 0
     - 1
   * - 101
     - 1
     - 3
   * - 110
     - 2
     - 0
   * - 111
     - 3
     - 2

The parity model  strategy is based on the paper
`"Quantum Policy Gradient Algorithm with Optimized Action Decoding" by Meyer et al.
<https://arxiv.org/abs/2212.06663v1>`_.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
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


class ProbabilityModel(QModel, ABC):
    r"""Model using angle encoding and probability values.

    This model implements a quantum unitary transformation U(x) composed of trainable
    and encoding layers, where depending on decoding strategy, measured bit strings are
    converted to output features.

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
        r"""Init :py:class:`ProbabilityModel`.

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
                used for computations. This includes the name of a PennyLane device
                and optional configuration settings.
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
        r"""Maps input ``X`` in the range ``min_max`` to $(-\pi, \pi]$.

        Args:
            X: input data with shape (`n_samples`, `n_features`).
            min_max: For each feature, the minimum and maximum value in its range.
        """
        # Coerce into an ndarray
        X = np.asarray(X)

        # Convert to angles between -pi and pi
        angles = 2 * np.pi * (X - min_max[0]) / (min_max[1] - min_max[0]) - np.pi

        # Set number of qubits required by model and validate
        self.n_qubits = angles.shape[1]
        if self.n_classes > 2**self.n_qubits:
            error_msg = (
                "The number of classes should be less than or equal to 2 to the power"
                "of the number of features."
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
    ) -> pennylane.measurements.MeasurementProcess:
        """Create modelled circuit.

        Args:
            weights: weights for trainable circuits
            x: data for the encoding circuit

        Returns:
            Measurements (probabilities) if the number of classes
            is greater than 2; otherwise measurements (expected values)
            are returned.
        """
        # Define circuit
        for w in weights[:-1, :, :, :]:
            self._W(w)
            self._S(x)
        self._W(weights[-1, :, :, :])

        # If there are more than 2 classes, measure probabilities
        if self.n_classes > 2:
            return pennylane.probs(wires=range(self.n_qubits))

        # Else measure the expected value for the first qubit
        return pennylane.expval(pennylane.PauliZ(0))

    def get_qfunc(self) -> Callable[[TensorType, TensorType], TensorType]:  # noqa: D102
        dev = self.backend.get_instance(wires=self.n_qubits)
        qnode = pennylane.QNode(self._circuit, dev, interface="torch")
        return partial(self._process_measurement, qnode=qnode, n_classes=self.n_classes)

    @staticmethod
    @abstractmethod
    def _process_measurement(
        weights: TensorType,
        x: TensorType,
        qnode: pennylane.QNode,
        n_classes: int,
    ) -> TensorType:
        """Define post-processing strategy."""


class ModuloModel(ProbabilityModel):
    r"""Model that uses modulo reduction for post-processing.

    The post-processing strategy assigns a class to an $n$-bit string
    $b$ according to the following formula:

    .. math::
        f(b) = \left[b\right]_{10} \mod M

    where:

        - $M$ is the number of classes,
        - $[\cdot]_{10}$ is the decimal representation of the argument.
    """

    @staticmethod
    def _process_measurement(
        weights: TensorType,
        x: TensorType,
        qnode: pennylane.QNode,
        n_classes: int,
    ) -> TensorType:
        """Define post-processing strategy."""
        measurement: TensorType = qnode(weights, x)
        if n_classes > 2:
            aggregated_measurement = [
                measurement[class_id::n_classes][
                    : measurement.numel() // n_classes
                ].sum()
                for class_id in range(n_classes)
            ]
            return torch.stack(aggregated_measurement).squeeze()
        return measurement


class ParityModel(ProbabilityModel):
    r"""Model that uses parity for post-processing.

    The post-processing strategy assigns a class to an $n$-bit string $b$
    according to the following formula:

    .. math::
        f(b) = \left[b_0 ... b_{m-2}\left(\bigoplus_{i=m-1}^{n-1} b_i\right) \right]_{10}

    where:

        - $m=\lceil \log_2(M) \rceil$ with $M$ being the number of classes,
        - $n$ is the number of bits,
        - $[\cdot]_{10}$ is the decimal representation of the argument.

    Reference: `"Quantum Policy Gradient Algorithm with Optimized Action Decoding"
    by Meyer et al. <https://arxiv.org/abs/2212.06663v1>`_
    """  # noqa : E501

    @staticmethod
    def _f(idx: int, n_bits_in: int, n_bits_out: int) -> int:
        """Post-processing function.

        Assigns a class index (an integer) to an arbitrary integer
        (corresponding to a measured bit string).

        Args:
            idx: state index.
            n_bits_in: number of bits used for the input index.
            n_bits_out: number of bits to be used for the output class index.

        Returns:
            Class index assigned.
        """
        idx_bit_array = np.array(
            list(map(int, [*np.binary_repr(idx, width=n_bits_in)]))
        )
        class_bit_array = idx_bit_array[-n_bits_out:]
        if n_bits_out < n_bits_in:
            class_bit_array[0] = idx_bit_array[: -(n_bits_out - 1)].sum() % 2

        return int("".join([str(elem) for elem in np.flip(class_bit_array)]), 2)

    @staticmethod
    def _process_measurement(
        weights: TensorType,
        x: TensorType,
        qnode: pennylane.QNode,
        n_classes: int,
    ) -> TensorType:
        """Define post-processing strategy."""
        measurement: TensorType = qnode(weights, x)
        if n_classes > 2:
            n_bits_in = int(np.log2(measurement.numel()))
            n_bits_out = int(np.ceil(np.log2(n_classes)))
            aggregated_measurement = [
                torch.zeros(1, requires_grad=False) for _ in range(n_classes)
            ]
            for idx, prob in enumerate(measurement):
                class_id = ParityModel._f(idx, n_bits_in, n_bits_out)
                if class_id >= n_classes:
                    continue
                aggregated_measurement[class_id] = (
                    aggregated_measurement[class_id] + prob
                )
            return torch.stack(aggregated_measurement).squeeze()
        return measurement
