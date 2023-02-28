"""
Basic utility functions used in the Variational Classifier.
"""

import numpy as np
from numpy.typing import NDArray


def get_bin(x: int, n: int) -> NDArray[np.uint8]:
    """Get binary string representation for integer

    Args:
        x: Input integer
        n: Total number of bits

    Raises:
        ValueError in case not enough bits are provided to represent x.

    Returns:
        bit array
    """
    if x.bit_length() > n:
        raise ValueError("More bits are needed to represent x.")
    return np.array([_ for _ in np.binary_repr(x, width=n)], dtype=np.uint8)


def get_decimal(x: NDArray[np.uint8]) -> np.int_:
    """
    Get decimal representation of bit array.

    Args:
        x: Binary bit array

    Returns:
        Integer representation
    """
    x = np.asarray(x)
    return np.flip(x) @ (1 << np.arange(len(x)))
