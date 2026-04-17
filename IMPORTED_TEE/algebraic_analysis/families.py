"""Fringe algebra families for ADEV2.

The goal is not to enumerate ordinary semisimple or Lie families again. These
generators target stranger corners suggested by `100 ideas.md`: square-zero
garbage layers, one-sided zero divisors, twisted composition laws, and signed
path-like algebras.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ade.core.algebra_state import AlgebraState


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    params: dict[str, Any]
    idea_refs: tuple[int, ...]

    @property
    def label(self) -> str:
        params = ",".join(f"{key}={value}" for key, value in sorted(self.params.items()))
        return f"{self.family}({params})"


def _set_annihilator_block(algebra: AlgebraState, start: int, width: int) -> None:
    for row in range(start, start + width):
        for col in range(algebra.n):
            for out in range(algebra.n):
                algebra.set_C(row, col, out, 0.0)
                algebra.set_C(col, row, out, 0.0)


def build_one_sided_zero_divisor(width: int) -> AlgebraState:
    """Associative algebra with one-sided zero divisors.

    Basis:
      L_i, R_i for i=0..width-1, plus one annihilator element Z.

    Products:
      L_i * R_i = Z
      all other products = 0

    This is associative because every nonzero product lands in the annihilator.
    It is strongly noncommutative and exposes order-sensitive routing gadgets.
    """
    dim = 2 * width + 1
    algebra = AlgebraState(dim, "R", f"ADEV2_one_sided_w{width}")
    z_idx = dim - 1
    for idx in range(width):
        left_idx = idx
        right_idx = width + idx
        algebra.set_C(left_idx, right_idx, z_idx, 1.0)
    return algebra


def build_nil_shadow(width: int) -> AlgebraState:
    """Commutative square-to-shadow algebra with square-zero radical.

    Basis:
      X_i, E_i for i=0..width-1.

    Products:
      X_i * X_i = E_i
      X_i * X_{i+1} = E_i + E_{i+1}
      anything involving E_j on either side = 0

    This realizes the "nilpotent witness slot" / "cancellation reservoir"
    ideas as an exact algebra family.
    """
    dim = 2 * width
    algebra = AlgebraState(dim, "R", f"ADEV2_nil_shadow_w{width}")
    shadow_start = width
    for idx in range(width):
        algebra.set_C(idx, idx, shadow_start + idx, 1.0)
    for idx in range(width - 1):
        algebra.set_C(idx, idx + 1, shadow_start + idx, 1.0)
        algebra.set_C(idx, idx + 1, shadow_start + idx + 1, 1.0)
        algebra.set_C(idx + 1, idx, shadow_start + idx, 1.0)
        algebra.set_C(idx + 1, idx, shadow_start + idx + 1, 1.0)
    _set_annihilator_block(algebra, shadow_start, width)
    return algebra


def build_parity_shadow(width: int) -> AlgebraState:
        """Parity-shadow algebra inspired by idea 1.

        Basis:
            B_i, P_i for i=0..width-1.

        Products:
            B_i * B_i = B_i
            B_i * P_i = P_i
            P_i * B_i = -P_i
            all remaining products = 0

        The shadow layer keeps track of ordered interactions without adding a full
        second algebra layer.
        """
        dim = 2 * width
        algebra = AlgebraState(dim, "R", f"ADEV2_parity_shadow_w{width}")
        shadow_start = width
        for idx in range(width):
                algebra.set_C(idx, idx, idx, 1.0)
                algebra.set_C(idx, shadow_start + idx, shadow_start + idx, 1.0)
                algebra.set_C(shadow_start + idx, idx, shadow_start + idx, -1.0)
        return algebra


def build_coefficient_split(width: int) -> AlgebraState:
        """Asymmetric coefficient-splitting algebra inspired by idea 4.

        Basis:
            L_i, R_i, S_i for i=0..width-1.

        Products:
            L_i * L_i = L_i
            R_i * R_i = R_i
            L_i * R_i = S_i
            R_i * L_i = -S_i
            anything involving S_i on either side = 0

        This encodes ordered scalar pairs with distinct left/right multiplication.
        """
        dim = 3 * width
        algebra = AlgebraState(dim, "R", f"ADEV2_coefficient_split_w{width}")
        left_start = 0
        right_start = width
        shadow_start = 2 * width
        for idx in range(width):
                algebra.set_C(left_start + idx, left_start + idx, left_start + idx, 1.0)
                algebra.set_C(right_start + idx, right_start + idx, right_start + idx, 1.0)
                algebra.set_C(left_start + idx, right_start + idx, shadow_start + idx, 1.0)
                algebra.set_C(right_start + idx, left_start + idx, shadow_start + idx, -1.0)
        _set_annihilator_block(algebra, shadow_start, width)
        return algebra


def _bits(value: int, width: int) -> list[int]:
    return [(value >> idx) & 1 for idx in range(width)]


def _twist_sign(left: int, right: int, width: int, twist_mask: int) -> float:
    """Sign from an upper-triangular bilinear form over Z_2.

    `twist_mask` packs the bits for pairs (i,j) with i<j in lexicographic order.
    The resulting sign cocycle is associative because the exponent is bilinear.
    """
    left_bits = _bits(left, width)
    right_bits = _bits(right, width)
    parity = 0
    bit_index = 0
    for row in range(width):
        for col in range(row + 1, width):
            if (twist_mask >> bit_index) & 1:
                parity ^= left_bits[row] & right_bits[col]
            bit_index += 1
    return -1.0 if parity else 1.0


def build_twisted_group_algebra(width: int, twist_mask: int) -> AlgebraState:
    """Twisted real group algebra of (Z/2Z)^width with +-1 cocycle.

    Multiplication:
      e_a * e_b = sigma(a,b) e_{a xor b}

    This family is associative by construction but typically noncommutative
    whenever the cocycle is not symmetric.
    """
    dim = 1 << width
    algebra = AlgebraState(dim, "R", f"ADEV2_twisted_group_k{width}_m{twist_mask}")
    for left in range(dim):
        for right in range(dim):
            out = left ^ right
            algebra.set_C(left, right, out, _twist_sign(left, right, width, twist_mask))
    return algebra


def build_signed_path_algebra() -> AlgebraState:
    """Small signed path algebra on a 3-cycle with sinks for length-2 paths.

    Basis order:
      p0,p1,p2,a01,a12,a20,s_plus,s_minus

    The sinks absorb length-2 path compositions with opposite signs, providing
    a minimal exact model of "signed path / cancellation reservoir" behaviour.
    """
    algebra = AlgebraState(8, "R", "ADEV2_signed_path_cycle3")
    p0, p1, p2 = 0, 1, 2
    a01, a12, a20 = 3, 4, 5
    s_plus, s_minus = 6, 7

    for vertex in (p0, p1, p2):
        algebra.set_C(vertex, vertex, vertex, 1.0)

    algebra.set_C(p0, a01, a01, 1.0)
    algebra.set_C(a01, p1, a01, 1.0)
    algebra.set_C(p1, a12, a12, 1.0)
    algebra.set_C(a12, p2, a12, 1.0)
    algebra.set_C(p2, a20, a20, 1.0)
    algebra.set_C(a20, p0, a20, 1.0)

    algebra.set_C(a01, a12, s_plus, 1.0)
    algebra.set_C(a12, a20, s_minus, -1.0)
    algebra.set_C(a20, a01, s_plus, 1.0)

    _set_annihilator_block(algebra, s_plus, 2)
    return algebra


def build_radical_extension(base_width: int, radical_dim: int, seed: int) -> AlgebraState:
    """Square-zero radical extension of a twisted-group base.

    The base is a dim `2^base_width` twisted group algebra. A square-zero radical
    is attached and selected base-base products leak into radical slots. This is
    intentionally fringe and generally nonassociative.
    """
    base = build_twisted_group_algebra(base_width, twist_mask=(1 << max(base_width - 1, 0)) - 1)
    base_dim = base.n
    algebra = AlgebraState(base_dim + radical_dim, "R", f"ADEV2_radical_ext_b{base_width}_r{radical_dim}_s{seed}")
    algebra.set_C_dense(base.C())

    rng = np.random.default_rng(seed)
    for left in range(base_dim):
        for right in range(base_dim):
            if rng.random() < 0.25:
                out = base_dim + int(rng.integers(0, radical_dim))
                value = float(rng.choice([-1.0, 1.0]))
                algebra.set_C(left, right, out, value)
    _set_annihilator_block(algebra, base_dim, radical_dim)
    return algebra


def build_random_relation_algebra(dim: int, density: float, seed: int) -> AlgebraState:
    """Sparse random short-relation algebra inspired by ideas 30, 47, and 86.

    Entries are drawn from a tiny signed alphabet with many forced zeros. The
    resulting family is intentionally broad rather than canonical; it exists to
    create a larger exploratory corpus for ADEV2 scans.
    """
    algebra = AlgebraState(dim, "R", f"ADEV2_random_relations_d{dim}_p{density:.2f}_s{seed}")
    rng = np.random.default_rng(seed)
    values = np.array([-2.0, -1.0, 1.0, 2.0])
    for left in range(dim):
        for right in range(dim):
            if rng.random() >= density:
                continue
            fanout = 1 if rng.random() < 0.7 else 2
            outputs = rng.choice(dim, size=fanout, replace=False)
            for out in np.atleast_1d(outputs):
                algebra.set_C(left, right, int(out), float(rng.choice(values)))
    return algebra


def build_candidate(spec: CandidateSpec) -> AlgebraState:
    family = spec.family
    params = spec.params
    if family == "parity_shadow":
        return build_parity_shadow(int(params["width"]))
    if family == "coefficient_split":
        return build_coefficient_split(int(params["width"]))
    if family == "one_sided_zero_divisor":
        return build_one_sided_zero_divisor(int(params["width"]))
    if family == "nil_shadow":
        return build_nil_shadow(int(params["width"]))
    if family == "twisted_group":
        return build_twisted_group_algebra(int(params["width"]), int(params["twist_mask"]))
    if family == "signed_path":
        return build_signed_path_algebra()
    if family == "radical_extension":
        return build_radical_extension(int(params["base_width"]), int(params["radical_dim"]), int(params["seed"]))
    if family == "random_relations":
        return build_random_relation_algebra(int(params["dim"]), float(params["density"]), int(params["seed"]))
    raise ValueError(f"Unknown ADEV2 family: {family}")


def fixed_candidate_specs(max_dim: int = 9) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []

    for width in (2, 3, 4):
        dim = 2 * width
        if dim <= max_dim:
            specs.append(CandidateSpec("parity_shadow", {"width": width}, (1, 29, 55, 97)))

    for width in (2, 3):
        dim = 3 * width
        if dim <= max_dim:
            specs.append(CandidateSpec("coefficient_split", {"width": width}, (4, 8, 51)))

    for width in (2, 3, 4):
        dim = 2 * width + 1
        if dim <= max_dim:
            specs.append(CandidateSpec("one_sided_zero_divisor", {"width": width}, (13, 29, 55)))

    for width in (2, 3, 4):
        dim = 2 * width
        if dim <= max_dim:
            specs.append(CandidateSpec("nil_shadow", {"width": width}, (2, 28, 68, 97)))

    for width in (2, 3):
        dim = 1 << width
        if dim > max_dim:
            continue
        max_pairs = width * (width - 1) // 2
        for twist_mask in range(1, 1 << max_pairs):
            specs.append(CandidateSpec("twisted_group", {"width": width, "twist_mask": twist_mask}, (22, 39, 90, 95)))

    if 8 <= max_dim:
        specs.append(CandidateSpec("signed_path", {}, (10, 21, 45, 66)))

    for radical_dim in (1, 2):
        dim = (1 << 2) + radical_dim
        if dim > max_dim:
            continue
        for seed in (11, 29):
            specs.append(CandidateSpec("radical_extension", {"base_width": 2, "radical_dim": radical_dim, "seed": seed}, (3, 28, 68, 84)))

    return specs


def _random_radical_spec(max_dim: int, random_seed: int, sample_index: int) -> CandidateSpec | None:
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 1, sample_index]))
    radical_dim = int(rng.choice([1, 2, 3]))
    dim = (1 << 2) + radical_dim
    if dim > max_dim:
        return None
    seed = int(rng.integers(0, 10_000_000))
    return CandidateSpec("radical_extension", {"base_width": 2, "radical_dim": radical_dim, "seed": seed}, (3, 28, 68, 84, 86))


def _random_relation_spec(max_dim: int, random_seed: int, sample_index: int) -> CandidateSpec | None:
    relation_dims = [dim for dim in (4, 5, 6, 7, 8, 9) if dim <= max_dim]
    if not relation_dims:
        return None
    relation_densities = [0.04, 0.06, 0.08, 0.12]
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 2, sample_index]))
    dim = int(rng.choice(relation_dims))
    density = float(rng.choice(relation_densities))
    seed = int(rng.integers(0, 10_000_000))
    return CandidateSpec("random_relations", {"dim": dim, "density": density, "seed": seed}, (30, 47, 86))


def total_candidate_count(
    max_dim: int = 9,
    random_radical_samples: int = 0,
    random_relation_samples: int = 0,
    random_seed: int = 0,
) -> int:
    total = len(fixed_candidate_specs(max_dim=max_dim))
    total += sum(1 for idx in range(max(random_radical_samples, 0)) if _random_radical_spec(max_dim, random_seed, idx) is not None)
    total += sum(1 for idx in range(max(random_relation_samples, 0)) if _random_relation_spec(max_dim, random_seed, idx) is not None)
    return total


def build_candidate_specs_slab(
    max_dim: int = 9,
    random_radical_samples: int = 0,
    random_relation_samples: int = 0,
    random_seed: int = 0,
    offset: int = 0,
    max_count: int | None = None,
) -> list[CandidateSpec]:
    remaining = max_count if max_count is not None else 1_000_000_000
    if remaining <= 0:
        return []

    fixed = fixed_candidate_specs(max_dim=max_dim)
    result: list[CandidateSpec] = []
    cursor = int(offset)

    if cursor < len(fixed):
        take = min(remaining, len(fixed) - cursor)
        result.extend(fixed[cursor:cursor + take])
        cursor = 0
        remaining -= take
    else:
        cursor -= len(fixed)

    radical_index = 0
    while remaining > 0 and radical_index < max(random_radical_samples, 0):
        spec = _random_radical_spec(max_dim, random_seed, radical_index)
        radical_index += 1
        if spec is None:
            continue
        if cursor > 0:
            cursor -= 1
            continue
        result.append(spec)
        remaining -= 1

    relation_index = 0
    while remaining > 0 and relation_index < max(random_relation_samples, 0):
        spec = _random_relation_spec(max_dim, random_seed, relation_index)
        relation_index += 1
        if spec is None:
            continue
        if cursor > 0:
            cursor -= 1
            continue
        result.append(spec)
        remaining -= 1

    return result


def default_candidate_specs(
    max_dim: int = 9,
    random_radical_samples: int = 0,
    random_relation_samples: int = 0,
    random_seed: int = 0,
) -> list[CandidateSpec]:
    total = total_candidate_count(
        max_dim=max_dim,
        random_radical_samples=random_radical_samples,
        random_relation_samples=random_relation_samples,
        random_seed=random_seed,
    )
    return build_candidate_specs_slab(
        max_dim=max_dim,
        random_radical_samples=random_radical_samples,
        random_relation_samples=random_relation_samples,
        random_seed=random_seed,
        offset=0,
        max_count=total,
    )