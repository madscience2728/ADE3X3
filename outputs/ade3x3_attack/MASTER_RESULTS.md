# ADE3x3 Attack Route Results

Status for this run: Phases 1 through 5 executed in the repository-local attack tree.

Windows note: the requested `/mnt/user-data/outputs/ade3x3_attack/` path is not present in this workspace, so outputs were written to `outputs/ade3x3_attack/` in the repository.

## Phase 1 Summary

- AlphaTensor public rank-23 passes the collection-level delta-in-eta test exactly: `rank(H)=14`, `rank(Delta)=10`, `rank([H|Delta])=14`.
- The exact projection `Delta = H * M` exists with a sparse integer matrix `M`; all nonzero coefficients are `+1` or `-1`.
- Single-term constant containment fails for all 54 delta coordinates.
- Single-term generic local dimensions are `rank J_eta = 15` and `rank J_(eta,delta) = 17`, so single-term containment also fails infinitesimally.

## Phase 2 Summary

- The Fourier identities linking `(eta1, eta2)` and `(xi+, xi-)` are correct.
- For real factors, `xi- = conjugate(xi+)` is correct.
- The proposed parity theorem is false.
- AlphaTensor itself gives `rank(H)=14` but `rank_C(Xi+)=9`, so `rank(H) != 2 * rank_C(Xi+)`.
- Therefore no valid parity tightening of the Step 52 nuisance budget is obtained from this route.

## Phase 3 Summary

- The full 216-element compatible symmetry orbit of the public AlphaTensor decomposition was generated and profiled exactly.
- All 216 symmetry-orbit decompositions satisfy `delta subset eta` in the fixed Step 51 basis.
- Across that exact symmetry orbit, `rank(H)` stays fixed at 14.
- The exact projection sparsity is not literally constant across the orbit in the fixed coordinate basis; in this run it ranged from 42 to 80 nonzero entries, and all exact projection coefficients remained integral.
- No exact non-orbit rank-23 decomposition was found in the perturbation pass (512 attempts total across sigma = 0.01, 0.05, 0.1, 0.5).
- No exact non-orbit rank-23 decomposition was found in the cold random rank-23 search (512 restarts).
- Phase 3 verdict: `delta subset eta` is universal across the AlphaTensor symmetry orbit, but universality beyond that equivalence class remains unclear.

## Phase 4 Summary

- The rank-search pass was run in the full tensor formulation, not the quotient-only formulation, because Phase 3 did not produce any exact non-orbit evidence supporting universal `delta subset eta`.
- Total candidate runs: 592.
- No exact rank-22 decomposition was found.
- The best rank-22 candidate came from the border-style vanishing-tail strategy with max-abs residual about `1.292e-3`.
- The best rank-22 candidate still had `rank(H)=18`, `rank(Nuisance)=22`, and quotient gain 0, so it showed no useful nuisance compression.
- Cold-start landscape summary from this run:
	- `R=21`: best max-abs residual about `1.157e-1`
	- `R=22`: best cold max-abs residual about `2.029e-2`
	- `R=23`: best cold max-abs residual about `1.107e-2`

## Phase 5 Summary

- Trajectory diagnostics: in the original term order, `rank(H)` first reaches 14 at `k=18`, while quotient gain 9 appears only at the full `k=23`; across 20 random orderings, the first hit of `rank(H)=14` ranged from `k=14` to `k=21`.
- AlphaTensor anisotropy identities: `H` has exact nullity 4, with four primitive integer relations among the 18 anisotropy coordinates.
- Projection matrix diagnostics: the exact AlphaTensor `M` has shape `18 x 54`, exact rank 10, nullity 44, coefficient alphabet `{+1,-1}`, and no trivial one-factor Kronecker collapse under the coarse reshapes tested.
- Pairwise interaction matrix: all 506 ordered pairs have pairwise anisotropy compression 0 and nuisance compression 0, so the compression is not explained by special 2-term building blocks.
- Standard 27-term baseline: verifies exactly with `rank(H)=18`, `rank(Delta)=0`, `rank(Nuisance)=18`, and quotient gain 9.

## Overall Picture

The strongest positive statement remains decomposition-specific but now has an exact orbit census behind it: `delta subset eta` is stable across the entire AlphaTensor symmetry orbit. The strongest negative statement is that this run found no exact non-orbit rank-23 decomposition carrying the same structure, and no exact rank-22 decomposition. Structurally, AlphaTensor's rank deficiency is enforced by four exact anisotropy identities and appears to be a genuinely collective phenomenon rather than a pairwise one.