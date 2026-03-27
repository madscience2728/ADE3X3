# Orbit Metadata Cache Repair Report

Generated: 2026-03-27 16:40:03

## Bug Description

Step10 orbit_metadata keyed records by orbit_id (sequential 0..N-1)
and used that orbit_id to index into the configs list.
Since orbit_id != canonical representative config_id (except by
coincidence for the first few orbits), every signature was computed
from the wrong representative config.

## Schemas Affected

All schemas were affected, but only some showed changed signature
counts: XX, CX, XC, CC, AX. BX and CXC happened to match because
the first N configs aligned with orbit representatives.

## Comparison Table

| Schema | Orbits | Old Sigs | Repaired Sigs | Changed | Orbit Complete |
|--------|--------|----------|---------------|---------|----------------|
| XX | 56 | 20 | 48 | yes | no |
| CX | 8 | 4 | 8 | yes | yes |
| XC | 8 | 4 | 8 | yes | yes |
| CC | 4 | 3 | 4 | yes | yes |
| AX | 10 | 3 | 8 | yes | no |
| BX | 10 | 8 | 8 | no | no |
| CXC | 50 | 50 | 50 | no | yes |

## Repair Confirmation

Repaired metadata now uses actual canonical orbit representatives
(lexicographic minimum over group images) for every orbit record.
Signatures are computed from the correct representative config.

## Verdict

Old cached signature counts are superseded.
Repaired counts should now be treated as current warehouse truth.
