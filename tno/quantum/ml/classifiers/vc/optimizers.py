"""This module is used to define optimizers for training quantum models."""
from typing import Any, Iterable, Mapping

from torch.optim import SGD, Adam, Optimizer
from torchtyping import TensorType


def get_optimizer(
    parameters: Iterable[TensorType], optimizer: Mapping[str, Any]
) -> Optimizer:
    """Create instance of an optimizer.

    Args:
        parameters: Tensor objects to optimize over.
        optimizer: Mapping of the form ``{"name": str, "options": dict}``.
          The value for ``"name"`` is used to determine the optimizer class.
          The value for ``"options"`` should be a ``dict`` and is passed
          as kwargs to the constructor of the corresponding optimizer
          class. The supported optimizers are:

          - SDG:
              - name: ``"stochastic_gradient_descent"``
              - options: see `SDG kwargs`__

          - Adam
              - name: ``"adam"``
              - options: see `Adam kwargs`__

    __ https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    __ https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    Returns:
        Instance of a torch optimizer.
    """
    optimizers = {
        "stochastic_gradient_descent": SGD,
        "adam": Adam,
    }
    if optimizer["name"] not in optimizers:
        raise ValueError("Invalid optimizer name.")

    optimizer_class = optimizers[optimizer["name"]]
    optimizer_instance: Optimizer
    optimizer_instance = optimizer_class(parameters, **optimizer["options"])
    return optimizer_instance
