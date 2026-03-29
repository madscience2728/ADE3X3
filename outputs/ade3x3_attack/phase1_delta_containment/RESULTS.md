# Phase 1: Delta-in-Eta Containment

Windows note: the prompt requested `/mnt/user-data/outputs/ade3x3_attack/`, but this workspace runs on Windows, so all outputs for this run were written under `outputs/ade3x3_attack/` inside the repository.

## Verdicts

- AlphaTensor collection-level containment: YES.
- Exact projection `Delta = H * M` with `H = [Eta1|Eta2]`: YES.
- Projection coefficients rational: YES, and stronger: all nonzero coefficients are integers in `{+1,-1}`.
- Single-term constant containment `delta = M * eta` with constant coefficients: NO.
- Single-term local-image containment: NO.

## AlphaTensor Empirical Test

Using the exact public 23-term coefficient table from Step 63 with gamma orientation `transpose`:

| matrix | exact rank | SVD rank @ 1e-10 | boundary singular value | next singular value | gap ratio |
|--------|------------|------------------|-------------------------|---------------------|-----------|
| `H = [Eta1|Eta2]` | 14 | 14 | 0.6245923009701738 | 3.635613436945375e-16 | 1.7179832559287528e+15 |
| `Delta` | 10 | 10 | 1.0347305722615396 | 3.965306018970992e-16 | 2.609459565821997e+15 |
| `[H|Delta]` | 14 | 14 | 1.0938551057348913 | 1.1325921217417934e-15 | 9.657979114782035e+14 |

So the collection-level test lands exactly at

`rank([H|Delta]) = rank(H) = 14`.

This confirms that for the public AlphaTensor rank-23 decomposition, every dead-X column lies in the column span of the anisotropy block.

## Exact Projection Matrix

The exact projection was built by taking the pivot eta columns

`[0,1,2,3,5,6,9,10,11,12,13,14,15,17]`

as a basis of `H` and setting nonpivot eta coefficients to zero. With that convention:

- `Delta = H * M` holds exactly over `Z`.
- `M` has only 46 nonzero entries across the full `18 x 54` matrix.
- Every nonzero coefficient is `+1` or `-1`.
- 28 of the 54 delta columns are identically zero for this decomposition.

Representative identities from `delta_projection_matrix.csv`:

- `delta[0,0,1,1] = eta1[0,1] + eta1[0,2]`
- `delta[0,0,2,1] = eta1[0,1] + eta1[0,2] + eta2[1,1]`
- `delta[0,1,0,1] = eta1[0,0] + eta1[0,2]`
- `delta[0,1,0,2] = -eta1[0,0] - eta1[0,2]`
- `delta[0,2,1,0] = -eta2[0,0]`

## Single-Term Symbolic Result

The single-term constant-coefficient question was tested exactly in the 81-monomial basis

`a[r,s] * b[t,u]`.

Each eta coordinate is supported only on live monomials with `s=t`, while each delta coordinate is a single dead monomial with `s!=t`. Therefore a dead monomial cannot be produced by any constant linear combination of eta coordinates. The exact linear solve confirms this for all 54 delta coordinates:

- Solvable delta coordinates: 0 / 54.

## Single-Term Jacobian Dimensions

The symbolic Jacobians were evaluated at five exact integer specializations. The ranks were stable across all samples:

- `rank J_eta = 15`
- `rank J_(eta,delta) = 17`

These values are already the structural ceilings:

- `eta` depends on the three channelwise rank-1 live slices `(a[:,s], b[s,:])`, each of dimension 5, so `rank J_eta <= 15`.
- `(eta,delta)` factors through the Segre map `(a,b) -> vec(a) tensor vec(b)`, whose image has generic dimension 17 because of the global rescaling `(a,b) -> (lambda a, lambda^-1 b)`.

So the generic single-term ranks are exactly 15 and 17. Since the joint image dimension is strictly larger than the eta-only image dimension, single-term containment fails even infinitesimally.

## Bottom Line

The Phase 1 result cleanly splits into two levels:

- Single-term containment fails.
- Collection-level containment holds for the public AlphaTensor rank-23 decomposition, and it holds with an exact sparse integer projection.

## Generated Files

- `test_containment.py`
- `symbolic_analysis.py`
- `rank_summary.csv`
- `delta_projection_matrix.csv`
- `containment_summary.json`
- `single_term_constant_containment.csv`
- `structural_dependency_map.csv`
- `jacobian_rank_samples.csv`
- `symbolic_summary.json`