"""Evidence labels for one fixed classical spiral-inflation relaxation."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpiralResult:
    """Incompatibility, relaxation feasibility, or unresolved numerical evidence.

    Even an exact primal certificate proves only relaxation feasibility, never
    membership in the original triangle model. Only verified dual contradictions
    use status ``incompatible``. No result applies to quantum source fanout.
    """
    status: str
    guarantee: str
    diagnostics: dict = field(default_factory=dict)
    certificate: object | None = None
    method: str = "triangle_spiral"
    resource: str = "classical_fanout"
