# ED Degree Computation

This directory is a reproducible Julia scaffold for the ED-degree task described in the workspace conversation.

What is implemented:

- Orbit/rank metadata for `R in {13, 19, 20, 21, 22, 23, 27}`.
- Exact construction of the `3 x 3` matrix multiplication tensor and seeded generic targets.
- A sequential runner that writes one JSON file per rank into `results/`.
- Placeholder outputs for `R=23` and `R=27` with `EDdeg = 1`, `best_frobenius = 0.0`.
- Explicit blocker reporting for `R=13, 19, 20, 21, 22`.

Why the nontrivial ranks are blocked:

- The repository documents that the legacy stabilizer-averaged or seed-transport symmetric family is algebraically the wrong family for exact `3 x 3` matrix multiplication. See:
  - `docs/SYMMETRY_CANON.md`
  - `docs/combinatorics_note.md`
  - `docs/gate_progress.md`
- The same documents say the correct next step is a parameterization with full stabilizer freedom for orbit members.
- That full-stabilizer polynomial chart is not implemented anywhere in the repo, and without it there is no defensible polynomial ED-critical system to send to HomotopyContinuation.jl.

This means the current scaffold is intentionally conservative: it does not fabricate a polynomial system for the wrong variety.

Files:

- `orbit_data.jl`: orbit configs, effective parameter counts, and rank metadata.
- `setup_system.jl`: tensor construction, generic target generation, and blocker text.
- `solve_rank.jl`: placeholder solve logic and per-rank JSON payload construction.
- `run_ed.jl`: main entry point; writes `results/ed_R*.json`.

Run:

```julia
julia --project=. run_ed.jl
```

Expected output:

- `results/ed_R13.json`
- `results/ed_R19.json`
- `results/ed_R20.json`
- `results/ed_R21.json`
- `results/ed_R22.json`
- `results/ed_R23.json`
- `results/ed_R27.json`

Notes:

- `R=23` and `R=27` are emitted as requested placeholders.
- For `R=13, 19, 20, 21, 22`, the JSON includes a `status` field and a `note` that explains the missing mathematical chart.
- Once a full-stabilizer polynomial chart exists, `setup_system.jl` and `solve_rank.jl` are the intended insertion points for the actual HomotopyContinuation solve.