"""Small independent-source witnesses with explicit factor conventions."""
from fractions import Fraction

import numpy as np

from ..core.game import Scenario
from .classical import ClassicalNetworkStrategy
from .quantum import QuantumNetworkStrategy, QuantumSource
from .structure import SourceNetwork
from .witnesses import BILOCAL_SCENARIO


def triangle_xor_strategy():
    """Three independent uniform edge bits; each party outputs the XOR of its bits."""
    network = SourceNetwork(3, ((0, 1), (1, 2), (2, 0)))
    scenario = Scenario((1, 1, 1), (2, 2, 2))
    return ClassicalNetworkStrategy.from_response_rules(
        network, scenario, ((Fraction(1, 2), Fraction(1, 2)),) * 3,
        (lambda x, values, a: a == (values[0] ^ values[1]),) * 3)


def bilocal_entanglement_swapping():
    """Two independent Bell pairs and a central Bell-state measurement.

    Outer observables are (Z+(-1)^x X)/sqrt(2). Bob returns the eigenvalue bits
    of Z tensor Z and X tensor X, encoded b=2*b0+b1. This gives I=J=1/2.
    Reference: https://arxiv.org/abs/1112.4502.
    """
    network = SourceNetwork(3, ((0, 1), (1, 2)))
    phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    source = QuantumSource(np.outer(phi, phi.conj()), (2, 2))
    x, z = np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    outer = tuple(tuple((np.eye(2) + (-1)**a * (z + (-1)**setting * x) / np.sqrt(2)) / 2
                        for a in range(2)) for setting in range(2))
    zz, xx = np.kron(z, z), np.kron(x, x)
    central = tuple((np.eye(4) + (-1)**(b//2) * zz) @ (np.eye(4) + (-1)**(b % 2) * xx) / 4
                    for b in range(4))
    return QuantumNetworkStrategy(network, (source, source), (outer, (central,), outer))


def bilocal_deterministic_branches():
    """Two bilocal behaviors whose equally weighted mixture is not bilocal.

    Branch zero outputs a=c=0; branch one outputs a=x,c=z. Bob always outputs
    b=0. The branches yield (I,J)=(1,0) and (0,1). Their mixture would require
    a shared global selector and yields (I,J)=(1/2,1/2).
    """
    network = SourceNetwork(3, ((0, 1), (1, 2)))
    result = []
    for branch in (0, 1):
        def outer(x, values, a, branch=branch):
            return a == branch*x
        result.append(ClassicalNetworkStrategy.from_response_rules(
            network, BILOCAL_SCENARIO, ((1,), (1,)),
            (outer, lambda x, values, a: a == 0, outer)))
    return tuple(result)
