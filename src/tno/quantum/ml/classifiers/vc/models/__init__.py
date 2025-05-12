"""This module contains quantum models that can be used for the variational classifier.

A quantum model defines a quantum circuit ansatz and a strategy how to extract a class
assignment from its measurements.

Currently, three models are available:

* Based on computing expected values, the :py:class:`~tno.quantum.ml.classifiers.vc.models.expected_value_model.ExpectedValueModel`.
* Based on probabilities, the :py:class:`~tno.quantum.ml.classifiers.vc.models.probability_model.ModuloModel` and
  :py:class:`~tno.quantum.ml.classifiers.vc.models.probability_model.ParityModel`.

Usage
-----

Models can be specified to the `model` argument of the :py:class:`~tno.quantum.ml.classifiers.vc.VariationalClassifier`.
See below for an example how the ``"modulo_model"`` can be specified consisting of two
layers with two trainable sublayers.

>>> from tno.quantum.ml.classifiers.vc import VariationalClassifier
>>> from tno.quantum.ml.datasets import get_iris_dataset
>>>
>>> X_training, y_training, X_validation, y_validation = get_iris_dataset(
...     n_features=4, n_classes=3, random_seed=0
... )
>>> vc = VariationalClassifier(
...     model={
...         "name": "modulo_model",
...         "options": {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5},
...     },
... )
>>> vc = vc.fit(X_training, y_training)
>>> predictions_validation = vc.predict(X_validation)

Developing New Quantum Models
-----------------------------

New models can be implemented by extending from :py:class:`~tno.quantum.ml.classifiers.vc.models.QModel`.

"""  # noqa: E501

from tno.quantum.ml.classifiers.vc.models._base import ModelError, QModel

__all__ = [
    "ModelError",
    "QModel",
]
