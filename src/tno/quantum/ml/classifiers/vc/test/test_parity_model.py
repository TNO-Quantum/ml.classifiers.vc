"""This module contains tests for the ParityModel assignments."""

import numpy as np
import pytest

from tno.quantum.ml.classifiers.vc.models.probability_model import ParityModel


@pytest.mark.parametrize(
    ("n_bits_in", "n_bits_out", "assignments"),
    [
        (4, 2, [0, 2, 1, 3, 1, 3, 0, 2, 1, 3, 0, 2, 0, 2, 1, 3]),
        (
            5,
            3,
            [
                0,
                4,
                2,
                6,
                1,
                5,
                3,
                7,
                1,
                5,
                3,
                7,
                0,
                4,
                2,
                6,
                1,
                5,
                3,
                7,
                0,
                4,
                2,
                6,
                0,
                4,
                2,
                6,
                1,
                5,
                3,
                7,
            ],
        ),
    ],
)
def test_class_assignment_parity_model(
    n_bits_in: int, n_bits_out: int, assignments: list[int]
) -> None:
    r"""Test correct class assignment of the Parity model.

    That is, the following equation holds:

    .. math::
    f(b) = \left[b_0 ... b_{m-2}\left(\bigoplus_{i=m-1}^{n-1} b_i\right) \right]_{10}

    where:

        - $m=\lceil \log_2(M) \rceil$ with $M$ being the number of classes,
        - $n$ is the number of bits,
        - $[\cdot]_{10}$ is the decimal representation of the argument.
    """
    for idx, class_id in enumerate(assignments):
        assert ParityModel._f(idx, n_bits_in, n_bits_out) == class_id


def test_maximal_class_assignment_parity_model() -> None:
    """Check every class assignments correspond to a flip of the bitstring."""
    n_bits = 5
    for idx in range(int(2**n_bits)):
        assert (
            int(
                np.binary_repr(ParityModel._f(idx, n_bits, n_bits), width=n_bits)[::-1],
                2,
            )
            == idx
        )
