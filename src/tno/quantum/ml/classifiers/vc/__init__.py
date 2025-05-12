"""This module implements a variational quantum classifier.

The classifier is implemented to be consistent with the
`scikit-learn estimator API <https://scikit-learn.org/stable/developers/develop.html>`_,
which means that the classifier can be used as any other (binary and multiclass) scikit-learn classifier and
combined with transforms through `Pipelines <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.
In addition, the :py:class:`~tno.quantum.ml.classifiers.vc.VariationalClassifier` makes use
of `PyTorch <https://pytorch.org/docs/stable/tensors.html>`_ tensors, optimizers, and loss functions.

Basic usage
-----------

>>> from tno.quantum.ml.classifiers.vc import VariationalClassifier
>>> from tno.quantum.ml.datasets import get_iris_dataset
>>> X_training, y_training, X_validation, y_validation = get_iris_dataset()
>>> vc = VariationalClassifier().fit(X_training, y_training, n_iter=5)
>>> predictions = vc.predict(X_validation)

Backends
--------

Internally, the :py:class:`~tno.quantum.ml.classifiers.vc.VariationalClassifier` class makes use of hybrid classical/quantum
models to represent a function to be learned. These quantum models give rise to circuits
which can be run on different backends, such as quantum hardware and quantum simulators.

Backends are selected by closely following the `PennyLane <https://pennylane.ai/>`_
interface for specifying
`devices <https://docs.pennylane.ai/en/stable/introduction/circuits.html#defining-a-device>`_.
As a result, `PennyLane devices <https://pennylane.ai/devices/>`_ are also supported.

Here's an example showcasing how `Quantum Inspire <https://www.quantum-inspire.com/>`_
hardware can be used through the
`PennyLane-QuantumInspire <https://github.com/QuTech-Delft/pennylane-quantuminspire>`_
plugin (note that you will need to install the plugin as described
`here <https://qutech-delft.github.io/pennylane-quantuminspire/>`_):

>>> from tno.quantum.ml.classifiers.vc import VariationalClassifier
>>> from tno.quantum.ml.datasets import get_wine_dataset
>>> X_training, y_training, X_validation, y_validation = get_wine_dataset(
...     n_features=5, n_classes=3, random_seed=0
... )
>>> vc = VariationalClassifier(
...     backend={
...         "name": "quantuminspire.qi",
...         "options": {"backend": "Starmon-5", "project": "my project"},
...     },
... )
>>> vc = vc.fit(X_training, y_training)  # doctest: +SKIP
>>> predictions_validation = vc.predict(X_validation)  # doctest: +SKIP

As an alternative example, here's how the ``default.qubit`` simulator provided
by PennyLane can be specified by the `backend` keyword:

.. code-block:: python

    backend={"name": "default.qubit", "options": {}}
"""  # noqa: E501

from tno.quantum.ml.classifiers.vc._variational_classifier import (
    ModelConfig,
    VariationalClassifier,
)

__all__ = ["ModelConfig", "VariationalClassifier"]
__version__ = "3.0.1"
