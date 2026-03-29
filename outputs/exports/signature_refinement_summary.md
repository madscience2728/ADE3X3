# Signature Collision Refinement Summary

**Generated:** 2026-03-27 22:55:38
**Script:** ade3x3_step33_refine_signature_collisions.py
**Provenance:** [EXACT_DERIVED] / [POST-REFINEMENT]

## Overview

This document records the minimal additional Boolean features needed to resolve
all signature collisions in the XX, AX, and BX schemas. The base signatures
(from step19) had several collision groups where multiple orbits shared the
same signature. Refinement features were identified to separate these orbits.

## XX Schema Refinement

**Base State:**
- 56 orbits total
- 48 distinct base signatures
- 4 collision groups (8 orbits involved)

**Collision Groups (Base Signature → Orbit IDs):**
1. `(False, False, True, False, False, True, False, False)` → orbits {26, 30, 32}
2. `(False, False, True, False, False, False, False, False)` → orbits {27, 31, 33}
3. `(False, False, False, False, False, True, False, False)` → orbits {44, 48, 50}
4. `(False, False, False, False, False, False, False, False)` → orbits {45, 49, 51}

**Refinement Features:**
```
(s2, t2, u2, r2)
```

Where for XX[X[r1,s1|t1,u1], X[r2,s2|t2,u2]]:
- `s2`: column index of second X atom's left part (∈ {0,1,2})
- `t2`: row index of second X atom's right part (∈ {0,1,2})
- `u2`: column index of second X atom's right part (∈ {0,1,2})
- `r2`: row index of second X atom's left part (∈ {0,1,2})

**Result:**
- 56 distinct refined signatures
- All collisions resolved ✓

**Collision Analysis:**
All collision groups shared the pattern `X[0,0|1,0]` as first atom, varying only
in the second atom's coordinates. The key discriminating features were `s2` and `t2`,
which capture which specific X atom appears in the second position.

## AX Schema Refinement

**Base State:**
- 10 orbits total
- 8 distinct base signatures
- 2 collision groups (4 orbits involved)

**Collision Groups (Base Signature → Orbit IDs):**
1. `(True, False, False)` → orbits {2, 4}
   - Orbit 2: `AX[A[0,0],X[0,1|0,0]]` (t=0)
   - Orbit 4: `AX[A[0,0],X[0,1|2,0]]` (t=2)
2. `(False, False, False)` → orbits {7, 9}
   - Orbit 7: `AX[A[0,0],X[1,1|0,0]]` (t=0)
   - Orbit 9: `AX[A[0,0],X[1,1|2,0]]` (t=2)

**Refinement Features:**
```
(t, u)
```

Where for AX[A[r_a,s_a], X[r,s|t,u]]:
- `t`: row index of X atom's right part (∈ {0,1,2})
- `u`: column index of X atom's right part (∈ {0,1,2})

**Result:**
- 10 distinct refined signatures
- All collisions resolved ✓

**Collision Analysis:**
Both collision groups differed only in the `t` coordinate (0 vs 2), representing
different row indices in the X atom's right part. The refinement feature `(t, u)`
fully separates all orbits.

## BX Schema Refinement

**Base State:**
- 10 orbits total
- 8 distinct base signatures
- 2 collision groups (4 orbits involved)

**Collision Groups (Base Signature → Orbit IDs):**
1. `(False, True, False)` → orbits {2, 8}
   - Orbit 2: `BX[B[0,0],X[0,0|1,0]]` (s=0)
   - Orbit 8: `BX[B[0,0],X[0,1|2,0]]` (s=1)
2. `(False, False, False)` → orbits {3, 9}
   - Orbit 3: `BX[B[0,0],X[0,0|1,1]]` (s=0)
   - Orbit 9: `BX[B[0,0],X[0,1|2,1]]` (s=1)

**Refinement Features:**
```
(s, r)
```

Where for BX[B[t_b,u_b], X[r,s|t,u]]:
- `s`: column index of X atom's left part (∈ {0,1,2})
- `r`: row index of X atom's left part (∈ {0,1,2})

**Result:**
- 10 distinct refined signatures
- All collisions resolved ✓

**Collision Analysis:**
Both collision groups differed only in the `s` coordinate (0 vs 1), representing
different column indices in the X atom's left part. The refinement feature `(s, r)`
fully separates all orbits.

## Refined Signature Format

For each schema, the complete refined signature is:

```
refined_signature = (base_signature, refinement_features)
```

This combined signature uniquely identifies each orbit.

## Exported Files

- `signatures_XX_refined.csv`: XX schema with refined signatures
- `signatures_AX_refined.csv`: AX schema with refined signatures
- `signatures_BX_refined.csv`: BX schema with refined signatures

Each CSV contains columns:
- `schema`: Schema name
- `orbit_id`: Orbit identifier (0-indexed)
- `rep_config_id`: Config ID of canonical representative
- `rep_readable`: Human-readable representative name
- `signature_key`: Base signature (original from step19)
- `orbit_size`: Number of configurations in orbit
- `stabilizer_size`: Size of orbit stabilizer subgroup
- `refined_signature`: Combined (base, refinement) signature
- `refinement_features`: Just the refinement features

## Verification

All refined signatures satisfy:
- orbit_size × stabilizer_size = 216 (group order)
- Each orbit has a unique refined signature
- Refinement features are minimal Boolean/ternary coordinates

## Principles Followed

1. **Minimality:** Refinement features use only the coordinates necessary to
   separate collision groups, not all available features.

2. **Ground Truth:** Features are exact coordinate values from the canonical
   orbit representatives, not derived or compressed.

3. **Explicit Encoding:** Ternary values (0,1,2) are recorded as-is, not
   collapsed to Boolean.

4. **Provenance:** All refinements are [EXACT_DERIVED] from the canonical
   orbit representatives computed in step18/19.
