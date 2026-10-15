"""
Body-Segment-Parameter (BSP) tables — SWAPPABLE.

Each table provides, per segment *type*:
    mass : segment mass as a fraction of total body mass (per instance)
    com  : longitudinal CoM location as a fraction of segment length,
           measured FROM the proximal endpoint toward the distal endpoint.

The segment *layout* (which landmarks bound each segment) lives in com.py and
is independent of the table, so switching tables is a one-flag change:

    from bsp_tables import get_table
    table = get_table("deleva", sex="male")   # or "dempster"

Primary table for this study: Dempster (1955) — cadaver-derived, no sex
variants. This is a deliberate choice (2026-08-21); see the endpoint caveat
below, which applies to it with particular force.

de Leva (1996), "Adjustments to Zatsiorsky-Seluyanov's segment inertia
parameters", J. Biomech. 29(9):1223-1230, is the swappable sensitivity-check
alternative. It is sex-specific, from living young adults, and its segment
endpoints are defined by JOINT CENTERS.

That last point is the one to keep in view: the segment layout in com.py is
built from pose keypoints, i.e. joint centres, which is the convention de Leva
is defined for and Dempster is not. Mixing a joint-centre layout with
cadaver-endpoint coefficients biases the result — mostly as a near-constant
offset, which largely cancels in a single-cam vs multi-cam AGREEMENT metric
(both sides use the same table) but does NOT cancel in any absolute CoM claim.
The trunk drives this: it is ~50% of body mass, and its com ratio differs
0.5000 (Dempster) vs 0.4486 (de Leva male), i.e. ~5% of trunk length.

------------------------------------------------------------------------------
!!! ACCURACY CAVEAT — READ BEFORE PUBLISHING !!!
The limb coefficients below are the standard, widely-cited de Leva / Dempster
values. The HEAD and TRUNK coefficients depend on how you define the segment
endpoints from the available keypoints (e.g. is "trunk proximal" the
suprasternale or the mid-shoulder?). The values here use the mid-shoulder /
mid-hip approximation appropriate for MediaPipe landmarks. Before you report
numbers in the paper, verify every coefficient against the ORIGINAL table for
your exact endpoint definitions. `validate_table()` checks that masses sum to
1.0; it CANNOT check that a coefficient is the right one.
------------------------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentParam:
    mass: float   # fraction of total body mass (per instance)
    com: float    # CoM as fraction of segment length, from proximal endpoint


# Segment types. Bilateral limbs (arm/forearm/hand/thigh/shank/foot) appear
# ONCE here but are instantiated TWICE (left+right) by the layout in com.py,
# so their `mass` is the per-side fraction.
SEGMENT_TYPES = (
    "head", "trunk",
    "upper_arm", "forearm", "hand",
    "thigh", "shank", "foot",
)

# Number of instances of each type in a whole body (for the mass-sum check).
_INSTANCES = {
    "head": 1, "trunk": 1,
    "upper_arm": 2, "forearm": 2, "hand": 2,
    "thigh": 2, "shank": 2, "foot": 2,
}


# --- de Leva (1996), Table 4 -------------------------------------------------
# mass = %BM/100 (per side for limbs); com = %segment length from proximal /100.
_DELEVA_MALE = {
    "head":      SegmentParam(0.0694, 0.5002),
    "trunk":     SegmentParam(0.4346, 0.4486),
    "upper_arm": SegmentParam(0.0271, 0.5772),
    "forearm":   SegmentParam(0.0162, 0.4574),
    "hand":      SegmentParam(0.0061, 0.7900),
    "thigh":     SegmentParam(0.1416, 0.4095),
    "shank":     SegmentParam(0.0433, 0.4459),
    "foot":      SegmentParam(0.0137, 0.4415),
}

_DELEVA_FEMALE = {
    "head":      SegmentParam(0.0668, 0.4841),
    "trunk":     SegmentParam(0.4257, 0.4151),
    "upper_arm": SegmentParam(0.0255, 0.5754),
    "forearm":   SegmentParam(0.0138, 0.4559),
    "hand":      SegmentParam(0.0056, 0.7474),
    "thigh":     SegmentParam(0.1478, 0.3612),
    "shank":     SegmentParam(0.0481, 0.4416),
    "foot":      SegmentParam(0.0129, 0.4014),
}


# --- Dempster (1955) — PRIMARY table for this study --------------------------
# Cadaver-derived (8 elderly male cadavers), no sex variants. CoM ratios are the
# commonly-cited Winter-textbook summaries; VERIFY before publishing. Note the
# suspiciously round 0.5000 head/trunk/foot ratios — those are the textbook
# roundings, and they are the coefficients most sensitive to endpoint choice.
_DEMPSTER = {
    "head":      SegmentParam(0.0810, 0.5000),
    "trunk":     SegmentParam(0.4970, 0.5000),
    "upper_arm": SegmentParam(0.0280, 0.4360),
    "forearm":   SegmentParam(0.0160, 0.4300),
    "hand":      SegmentParam(0.0060, 0.5060),
    "thigh":     SegmentParam(0.1000, 0.4330),
    "shank":     SegmentParam(0.0465, 0.4330),
    "foot":      SegmentParam(0.0145, 0.5000),
}


_REGISTRY = {
    ("deleva", "male"):   _DELEVA_MALE,
    ("deleva", "female"): _DELEVA_FEMALE,
    ("dempster", None):   _DEMPSTER,
    ("dempster", "male"): _DEMPSTER,
    ("dempster", "female"): _DEMPSTER,
}


def get_table(name: str = "deleva", sex: str | None = "male") -> dict[str, SegmentParam]:
    """Return a BSP table by name and (for de Leva) sex.

    name : "deleva" | "dempster"
    sex  : "male" | "female"  (ignored for dempster)
    """
    name = name.lower()
    sex = sex.lower() if sex else None
    key = (name, sex)
    if key not in _REGISTRY:
        if name == "deleva":
            raise ValueError("de Leva requires sex='male' or sex='female'")
        raise ValueError(
            f"Unknown BSP table {(name, sex)!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    table = _REGISTRY[key]
    validate_table(table, context=f"{name}/{sex}")
    return table


def validate_table(table: dict[str, SegmentParam], context: str = "", tol: float = 1e-3) -> float:
    """Assert whole-body mass fractions sum to 1.0. Returns the sum."""
    missing = set(SEGMENT_TYPES) - set(table)
    if missing:
        raise ValueError(f"[{context}] table missing segment types: {sorted(missing)}")
    total = sum(table[t].mass * _INSTANCES[t] for t in SEGMENT_TYPES)
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"[{context}] segment masses sum to {total:.5f}, expected 1.0 "
            f"(tolerance {tol})."
        )
    return total


def list_tables() -> list[tuple[str, str | None]]:
    return sorted(_REGISTRY, key=lambda k: (k[0], k[1] or ""))
