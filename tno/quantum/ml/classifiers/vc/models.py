"""This module is used to define quantum models.

To add a new model, you should implement the :py:class:`~vc.models.QModel`
interface and update :py:func:`~vc.models.get_model`.
"""
import copy
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Tuple, Type, Union

import numpy as np
import pennylane
import pennylane.measurements
import torch
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from torchtyping import TensorType

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=attribute-defined-outside-init


class ModelError(Exception):
    """Module exception."""

    def __init__(self, message: str):
        """Init ModelError."""
        super().__init__(message)


def _get_device(backend: Dict[str, Any], n_qubits: int) -> pennylane.QubitDevice:
    """Get qubit device using PennyLane.

    Args:
        backend: see docstring of :py:class:`~vc.models.QModel`.
        n_qubits: number of qubits.

    Returns:
        PennyLane device.
    """
    return pennylane.device(
        backend["name"], wires=n_qubits, **backend.get("options", {})
    )


class QModel(ABC):
    """Abstract base class for quantum models."""

    def __init__(self, backend: Dict[str, Any], n_classes: int) -> None:
        """
        Init QModel.

        Args:
            backend: dictionary of the form ``{"name": str, "options": dict}``,
              where the value for ``"name"`` is the name of a PennyLane device and
              the value for ``"options"`` is a dict to be passed as kwargs to PennyLane
              when creating the device. Example: ``{"name": "default.qubit", "options": {}}``
            n_classes: number of classes.
        """
        self.backend = backend
        self.n_classes = n_classes

    @abstractmethod
    def preprocess(
        self, X: ArrayLike, min_max: Tuple[float, float]
    ) -> NDArray[np.float_]:
        """Convert ``X`` to features. This function should set ``self.n_qubits``."""

    @abstractmethod
    def get_init_weights(
        self, random: bool, random_state: RandomState
    ) -> NDArray[np.float_]:
        """Generate weights to be used as initial trainable parameters."""

    @abstractmethod
    def get_qfunc(self) -> Callable[[TensorType, TensorType], TensorType]:
        """Generate and return a quantum function."""


class ExpectedValuesModel(QModel):
    """Model that uses angle encoding and expected values."""

    def __init__(
        self,
        backend: Dict[str, Union[str, Any]],
        n_classes: int,
        n_layers: int = 2,
        n_trainable_sublayers: int = 2,
        scaling: float = 0.5,
    ) -> None:
        r"""Init ExpectedValuesModel.

        This model implements a unitary $U(x)$ of the form:

        .. math::
            U(x) = W_{L + 1} \cdot S(x) \cdot W_{L} \cdot \ldots
            \cdot W_{2} \cdot S(x) \cdot W_{1}

        where:

            - $x$ is the data to encode,
            - $S$ is an encoding circuit,
            - $W$ is a trainable circuit.

        Args:
            backend: see docstring of :py:class:`~vc.models.QModel`.
            n_classes: see docstring of :py:class:`~vc.models.QModel`.
            n_layers: number of layers in $U(x)$ (equal to $L$).
            n_trainable_sublayers: number of layers for each $W$.
            scaling: scaling to apply to the preprocessed data, see
              :py:meth:`~vc.models.ExpectedValuesModel.preprocess`.
        """
        super().__init__(backend, n_classes)
        self.n_layers = n_layers
        self.n_trainable_sublayers = n_trainable_sublayers
        self.scaling = scaling

    def preprocess(
        self, X: ArrayLike, min_max: Tuple[float, float]
    ) -> NDArray[np.float_]:
        r"""Maps input ``X`` in the range ``min_max`` to $(-\pi, \pi]$.

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
            raise ModelError(
                "The number of classes should be less than or equal to the number "
                "of features."
            )

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
        self, random: bool, random_state: RandomState
    ) -> NDArray[np.float_]:
        r"""Get init weights between 0 and $2\pi$."""
        array_shape = self.n_layers + 1, self.n_trainable_sublayers, self.n_qubits, 3
        if random:
            return 2 * np.pi * random_state.random(size=array_shape)

        return np.zeros(array_shape)

    def _circuit(
        self,
        weights: TensorType,
        x: TensorType,
    ) -> Tuple[pennylane.measurements.MeasurementProcess, ...]:
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
        if self.n_classes > 2:
            wires = range(self.n_classes)
        else:
            wires = range(1)
        return tuple(pennylane.expval(pennylane.PauliZ(wire)) for wire in wires)

    def get_qfunc(self) -> Callable[[TensorType, TensorType], TensorType]:
        """Define callable based on circuit."""
        dev = _get_device(self.backend, self.n_qubits)
        qnode = pennylane.QNode(self._circuit, dev, interface="torch")

        def _process_measurement(
            weights: TensorType, x: TensorType, qnode: pennylane.QNode = qnode
        ) -> TensorType:
            return qnode(weights, x)

        return _process_measurement


class ProbabilitiesModel(QModel, ABC):
    """Generic model that uses probabilities."""

    def __init__(
        self,
        backend: Dict[str, Any],
        n_classes: int,
        n_layers: int = 2,
        n_trainable_sublayers: int = 2,
        scaling: float = 0.5,
    ) -> None:
        r"""Init ProbabilitiesModel.

        This model implements a unitary $U(x)$ of the form:

        .. math::
            U(x) = W_{L + 1} \cdot S(x) \cdot W_{L} \cdot \ldots
            \cdot W_{2} \cdot S(x) \cdot W_{1}

        where:

            - $x$ is the data to encode,
            - $S$ is an encoding circuit,
            - $W$ is a trainable circuit.

        Args:
            backend: see docstring of :py:class:`~vc.models.QModel`.
            n_classes: see docstring of :py:class:`~vc.models.QModel`.
            n_layers: number of layers in $U(x)$ (equal to $L$).
            n_trainable_sublayers: number of layers for each $W$.
            scaling: scaling to apply to the preprocessed data, see
                     :py:meth:`~vc.models.ProbabilitiesModel.preprocess`.
        """
        super().__init__(backend, n_classes)
        self.n_layers = n_layers
        self.n_trainable_sublayers = n_trainable_sublayers
        self.scaling = scaling

    def preprocess(
        self, X: ArrayLike, min_max: Tuple[float, float]
    ) -> NDArray[np.float_]:
        r"""Maps input ``X`` in the range ``min_max`` to $(-\pi, \pi]$.

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
        if self.n_classes > 2**self.n_qubits:
            raise ModelError(
                "The number of classes should be less than or equal to 2 to the power"
                "of the number of features."
            )

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
        self, random: bool, random_state: RandomState
    ) -> NDArray[np.float_]:
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

    def get_qfunc(self) -> Callable[[TensorType, TensorType], TensorType]:
        """Get callable based on circuit."""
        dev = _get_device(self.backend, self.n_qubits)
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


class ModuloModel(ProbabilitiesModel):
    r"""Model that uses post-processing with modulo.

    See docstring of :py:class:`.vc.models.ProbabilitiesModel`.

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
            aggregated_measurement = []
            for class_id in range(n_classes):
                aggregated_measurement.append(
                    measurement[class_id::n_classes][
                        : measurement.numel() // n_classes
                    ].sum()
                )
            return torch.stack(aggregated_measurement).squeeze()
        return measurement


class ParityModel(ProbabilitiesModel):
    r"""Model that uses parity post-processing.

    See docstring of :py:class:`.vc.models.ProbabilitiesModel`.

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
    """

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


def get_model(
    model: Dict[str, Any],
    backend: Dict[str, Any],
    n_classes: int,
) -> QModel:
    """Create instance of a quantum model.

    Args:
        model: dictionary of the form ``{"name": str, "options": dict}``.
          The value for ``"name"`` is used to determine the model class.
          The value for ``"options"`` should be a ``dict`` and is passed
          as kwargs to the constructor of the model class.
          Note: if there's a ``"backend"``
          key in the value for ``"options"``, it will be ignored.
        backend: see docstring of :py:class:`~vc.models.QModel`.
        n_classes: number of classes.

    Returns:
        An instance of a quantum model.
    """
    # Make sure there's no conflicting backend key
    model = copy.deepcopy(model)
    model.pop("backend", None)

    # Instantiate model
    models: Dict[str, Type[QModel]]
    models = {
        "expected_values_model": ExpectedValuesModel,
        "modulo_model": ModuloModel,
        "parity_model": ParityModel,
    }
    if model["name"] not in models:
        raise ValueError("Invalid model name.")

    model_class = models[model["name"]]
    model_instance = model_class(
        backend,
        n_classes,
        **model["options"],
    )
    return model_instance
