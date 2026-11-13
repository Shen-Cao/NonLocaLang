"""Small explicit inclusion rules for standard unrestricted correlation models."""
from ..core.resources import Resource, ResourceSpec

STANDARD_CHAIN = (Resource.LOCAL, Resource.QUANTUM_FINITE, Resource.QUANTUM_CLOSURE,
                  Resource.QUANTUM_COMMUTING, Resource.NONSIGNALLING)


def is_standard(resource):
    # Compare normalized dataclass fields, without serializing optional matrix data.
    return type(resource) is ResourceSpec and resource == ResourceSpec(resource.kind)


def includes(larger, smaller):
    """Set inclusion only; it must not be reused to infer attainment or membership."""
    return STANDARD_CHAIN.index(Resource(larger)) >= STANDARD_CHAIN.index(Resource(smaller))
