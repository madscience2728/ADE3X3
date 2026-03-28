# Step 70: Depth-2 Arithmetic Circuit Attack
Generated: 2026-03-28T19:39:38

[EXACT_DERIVED] + [MEASURED_FROM_CODE]

## Task 4: Recursive Strassen on 4x4 Padding

- Full recursive leaf count before pruning: 49
- Leaf count needed for the padded top-left 3x3 target: 31
- Exact top-left 3x3 verification after pruning: True
- Beats 23? False

## Task 6: Pair-Ratio Structure

- alphatensor23_constant_multiple_pair_count: 59
- alphatensor23_disjoint_support_pair_count: 191
- alphatensor23_unstructured_overlap_pair_count: 3
- strassen2x2_constant_multiple_pair_count: 12
- strassen2x2_disjoint_support_pair_count: 9
- strassen2x2_retained_missing_structured_pair_count: 7

## Task 3: 2x2 Baseline

- Structured retained-vs-missing Strassen pairs among {m1,m2,m3,m5} and {m4,m6,m7}: 7
- This does not reprove the depth-2 lower bound 7 for 2x2, but it does show there is no immediate constant-multiple or one-sided pair-ratio collapse inside the obvious Strassen split.

## Task 5: Pan Literature Status

- Pan_trilinear_aggregation_explicit_3x3_circuit: not_recovered_from_available_fetched_content | Fetched public summaries confirm Pan's 1978 aggregating/uniting/canceling line and his 1982 practical subcubic algorithm, but they did not provide an explicit small-3x3 depth-2 circuit or concrete multiplication count that could be instantiated directly in ADE3x3.

[INTERPRETATION]

The clean Step 70 result is that the most direct recursive depth-2 attack does not threaten rank 23: once 3x3 is zero-padded to 4x4 and Strassen is applied recursively, the exact leaf count needed to recover the top-left 3x3 block remains above 23 even after pruning leaf multiplications that are identically zero or only feed discarded padded outputs.
The AlphaTensor pair-ratio scan also does not reveal an obvious depth-2 factoring mechanism. Because every depth-1 term is already rank-1 in the A/B profile space, the only cheap pairwise collapses would come from proportional or one-sided ratios on common support. Step 70 records how often that happens exactly, and for the obvious 2x2 Strassen split it does not produce a direct 6-multiplication collapse.