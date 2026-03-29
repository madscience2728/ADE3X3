# Phase 33 Results

## 33a. Track A: 27-Symbol Encoding

[GROUND_TRUTH] The 27-symbol word is the per-term stack W_k = [P_0[k,:], P_1[k,:], P_2[k,:]] with P_s[k,(r,u)] = alpha_k[r,s] * beta_k[s,u].

[EXACT_DERIVED] For each known 3x3 decomposition, every cross-section block D_st was tested against span(H) with H = [K_0-K_1 | K_1-K_2].

### alphatensor_rank23

- rank(Gamma) = 9, dim ker(Gamma) = 14, rank(H) = 14.
- All six D_st blocks lie in span(H): True.
- All six D_st blocks lie in span([Sigma|H]): True.

| block | in span(H) | rank(M_st) | nnz(M_st) | density | coeffs | first 9-row block rank | second 9-row block rank |
|-------|------------|------------|-----------|---------|--------|------------------------|-------------------------|
| D_01 | True | 2 | 7 | 0.0432 | -1, 1 | 2 | 0 |
| D_02 | True | 3 | 12 | 0.0741 | -1, 1 | 3 | 3 |
| D_10 | True | 2 | 8 | 0.0494 | -1, 1 | 2 | 0 |
| D_12 | True | 2 | 5 | 0.0309 | -1, 1 | 1 | 1 |
| D_20 | True | 1 | 8 | 0.0494 | -1, 1 | 0 | 1 |
| D_21 | True | 2 | 6 | 0.0370 | -1, 1 | 0 | 2 |

### standard_rank27

- rank(Gamma) = 9, dim ker(Gamma) = 18, rank(H) = 18.
- All six D_st blocks lie in span(H): True.
- All six D_st blocks lie in span([Sigma|H]): True.

| block | in span(H) | rank(M_st) | nnz(M_st) | density | coeffs | first 9-row block rank | second 9-row block rank |
|-------|------------|------------|-----------|---------|--------|------------------------|-------------------------|
| D_01 | True | 0 | 0 | 0.0000 | none | 0 | 0 |
| D_02 | True | 0 | 0 | 0.0000 | none | 0 | 0 |
| D_10 | True | 0 | 0 | 0.0000 | none | 0 | 0 |
| D_12 | True | 0 | 0 | 0.0000 | none | 0 | 0 |
| D_20 | True | 0 | 0 | 0.0000 | none | 0 | 0 |
| D_21 | True | 0 | 0 | 0.0000 | none | 0 | 0 |

[MEASURED_FROM_CODE] Full exact H-recovery matrices M_st and [Sigma|H]-recovery matrices are exported in this session directory as JSONL artifacts.

## 33b. Track B: Clone-Frame Saturation

[EXACT_DERIVED] Each channel permutation pi in S_3 was tested via H_pi = [K_pi(0)-K_pi(1) | K_pi(1)-K_pi(2)].

### alphatensor_rank23

- dim ker(Gamma) = 14.
- Identity-frame rank(H) = 14, saturation = True.
- All clone frames saturate individually: True.
- Synthetic defective triples tested: 32 wildcard trials plus named structured cases; any factorized defective triple found = False.

| perm | rank(H_pi) | saturates ker(Gamma) |
|------|------------|----------------------|
| 012 | 14 | True |
| 021 | 14 | True |
| 102 | 14 | True |
| 120 | 14 | True |
| 201 | 14 | True |
| 210 | 14 | True |

| synthetic case | kind | rank(H) | defect | factorized rank-1 faces? |
|----------------|------|---------|--------|---------------------------|
| k0_equals_k1 | named | 9 | 5 | False |
| all_collinear | named | 9 | 5 | False |
| shared_two_plane | named | 2 | 12 | False |
| wildcard_trial_01 | wildcard | 1 | 13 | False |
| wildcard_trial_02 | wildcard | 1 | 13 | False |
| wildcard_trial_03 | wildcard | 3 | 11 | False |
| wildcard_trial_04 | wildcard | 3 | 11 | False |
| wildcard_trial_05 | wildcard | 1 | 13 | False |

[WILDCARD] Additional wildcard synthetic trials are exported in JSONL; the sample above shows the defect families actually tested.

### standard_rank27

- dim ker(Gamma) = 18.
- Identity-frame rank(H) = 18, saturation = True.
- All clone frames saturate individually: True.
- Synthetic defective triples tested: 32 wildcard trials plus named structured cases; any factorized defective triple found = False.

| perm | rank(H_pi) | saturates ker(Gamma) |
|------|------------|----------------------|
| 012 | 18 | True |
| 021 | 18 | True |
| 102 | 18 | True |
| 120 | 18 | True |
| 201 | 18 | True |
| 210 | 18 | True |

| synthetic case | kind | rank(H) | defect | factorized rank-1 faces? |
|----------------|------|---------|--------|---------------------------|
| k0_equals_k1 | named | 9 | 9 | False |
| all_collinear | named | 9 | 9 | False |
| shared_two_plane | named | 2 | 16 | False |
| wildcard_trial_01 | wildcard | 2 | 16 | False |
| wildcard_trial_02 | wildcard | 1 | 17 | False |
| wildcard_trial_03 | wildcard | 3 | 15 | False |
| wildcard_trial_04 | wildcard | 2 | 16 | False |
| wildcard_trial_05 | wildcard | 3 | 15 | False |

[WILDCARD] Additional wildcard synthetic trials are exported in JSONL; the sample above shows the defect families actually tested.

## 33c. Track C: Tiling-Preserving Rescaling Test

[EXACT_DERIVED] Under the tiling identities Gamma * P_s = I, the matched faces P_s are invariant while each D_st row is rescaled by lambda_s^(k) / lambda_t^(k). These rescalings preserve the 81 matched-face equations but generally change the full bilinear algorithm because they do not preserve the nuisance cancellation equations Gamma * H = 0 and Gamma * Delta = 0.

### alphatensor_rank23

- Observed gauge-orbit dimension across cross-sections: 28.
- Tangent escape from span(H) detected: True.
- Finite sampled gauge escape from span(H) detected: True.

| block | active rows | tangent escape? | sampled escape? |
|-------|-------------|-----------------|-----------------|
| D_01 | 7 | True | True |
| D_02 | 8 | True | True |
| D_10 | 7 | True | True |
| D_12 | 6 | True | True |
| D_20 | 4 | True | True |
| D_21 | 5 | True | True |

### standard_rank27

- Observed gauge-orbit dimension across cross-sections: 0.
- Tangent escape from span(H) detected: False.
- Finite sampled gauge escape from span(H) detected: False.

| block | active rows | tangent escape? | sampled escape? |
|-------|-------------|-----------------|-----------------|
| D_01 | 0 | False | False |
| D_02 | 0 | False | False |
| D_10 | 0 | False | False |
| D_12 | 0 | False | False |
| D_20 | 0 | False | False |
| D_21 | 0 | False | False |

## 33d. Track D: Khatri-Rao Absorption

[EXACT_DERIVED] The sanity-check question here is whether each D_st is recoverable from [Sigma|H], first in Strassen 2x2, then on the known 3x3 decompositions.

### strassen_2x2

- All D_st blocks recover from span(H): True.
- All D_st blocks recover from span([Sigma|H]): True.

| block | in span(H) | in span([Sigma|H]) | rank(recovery) | coeffs |
|-------|------------|--------------------|----------------|--------|
| D_01 | True | True | 1 | -1, 1 |
| D_10 | True | True | 1 | 1 |

### alphatensor_rank23

- All D_st blocks recover from span(H): True.
- All D_st blocks recover from span([Sigma|H]): True.

| block | in span(H) | in span([Sigma|H]) | rank(recovery) | coeffs |
|-------|------------|--------------------|----------------|--------|
| D_01 | True | True | 2 | -1, 1 |
| D_02 | True | True | 3 | -1, 1 |
| D_10 | True | True | 2 | -1, 1 |
| D_12 | True | True | 2 | -1, 1 |
| D_20 | True | True | 1 | -1, 1 |
| D_21 | True | True | 2 | -1, 1 |

### standard_rank27

- All D_st blocks recover from span(H): True.
- All D_st blocks recover from span([Sigma|H]): True.

| block | in span(H) | in span([Sigma|H]) | rank(recovery) | coeffs |
|-------|------------|--------------------|----------------|--------|
| D_01 | True | True | 0 | none |
| D_02 | True | True | 0 | none |
| D_10 | True | True | 0 | none |
| D_12 | True | True | 0 | none |
| D_20 | True | True | 0 | none |
| D_21 | True | True | 0 | none |

## 33e. Interpretation

[INTERPRETATION] Track A is the central positive result: on both known 3x3 decompositions, every individual cross-section block D_st is already exactly recoverable from H alone, not merely from [Sigma|H]. The resulting recovery matrices M_st are sparse, low-rank, and use only coefficients in {-1, 0, 1}, which points to combinatorial structure in the fiber coordinates rather than a numerical accident.

[INTERPRETATION] Track B gives a second positive result and one negative obstruction. Positive: every clone frame already saturates ker(Gamma) individually on AlphaTensor and the standard algorithm, so the union-of-clones argument is unnecessary on the known exact decompositions. Negative: in synthetic defective families inside ker(Gamma), the low-rank H defects are easy to manufacture at the right-inverse level but did not survive the sampled factorized rank-1 face test.

[INTERPRETATION] Track C is not an obstruction to Delta containment; it is a confirmation that the tiling-preserving rescalings are not true algorithm symmetries. They preserve Gamma * P_s = I but generally violate Gamma * H = 0 and Gamma * Delta = 0, so the observed escapes simply show that the 81 tiling equations alone are insufficient. The full 729 multiplication equations, and especially the 648 nuisance cancellation equations, are the constraints that must force containment.

[INTERPRETATION] Track D passes the mandatory Strassen sanity check and then matches Track A on the known 3x3 decompositions. The remaining gap is therefore not whether the known decompositions exhibit faithful encoding, but how to derive the observed M_st matrices directly from the fiber-mode definitions together with the full nuisance cancellation system.

[OPEN_FRONT] The concrete next gap for Phase 34 is now precise: start from the per-term identity D_st[k,:] = diag(beta_k[t,:] / beta_k[s,:]) * P_s[k,:], expand P_s = X_0 + K_s in the centered-lift basis, eliminate with K_0 + K_1 + K_2 = 0 and H = [K_0-K_1 | K_1-K_2], and then use the full nuisance cancellation equations Gamma * H = 0 and Gamma * Delta = 0 to show that the resulting expression linearizes to the sparse {-1, 0, 1} recovery matrices M_st by index structure alone.
