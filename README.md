# TNO Quantum: Variational classifier

TNO Quantum provides generic software components aimed at facilitating the development
of quantum applications.

The `tno.quantum.ml.classifiers.vc` package provides a `VariationalClassifier` class, which has been implemented 
in accordance with the
[scikit-learn estimator API](https://scikit-learn.org/stable/developers/develop.html).
This means that the classifier can be used as any other (binary and multiclass)
scikit-learn classifier and combined with transforms through
[Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
In addition, the `VariationalClassifier` makes use of
[PyTorch](https://pytorch.org/docs/stable/tensors.html) tensors, optimizers, and loss
functions.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*

## Documentation

Documentation of the `tno.quantum.ml.classifiers.vc` package can be found [here](https://tno-quantum.github.io/ml.classifiers.vc/).


## Install

Easily install the `tno.quantum.ml.classifiers.vc` package using pip:

```console
$ python -m pip install tno.quantum.ml.classifiers.vc
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.quantum.ml.classifiers.vc[tests]'
```

## Example

Here's an example of how the `VariationalClassifier` class can be used for
classification based on the
[Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set):
Note that `tno.quantum.ml.datasets` is required for this example.

```python
from tno.quantum.ml.classifiers.vc import VariationalClassifier
from tno.quantum.ml.datasets import get_iris_dataset

X_training, y_training, X_validation, y_validation = get_iris_dataset()
vc = VariationalClassifier()
vc = vc.fit(X_training, y_training)
predictions_validation = vc.predict(X_validation)
```
