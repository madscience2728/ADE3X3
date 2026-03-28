# Step 61: Nuisance-First Architecture
Generated: 2026-03-28T15:51:52

[EXACT_DERIVED]

Step 61 reverses the usual perspective. Instead of trying to eliminate nuisance, it shows exactly what the Step 51-52 framework forces at small ranks: zero nuisance is impossible below rank 27, and low-rank algorithms must engineer a highly compressed nuisance span inside the Gamma-nullspace.

R=9 status: impossible_by_zero_nuisance_contradiction
Dead-free global lower bound: 27
R=18 Gamma-nullity: 9
R=18 minimum dead dependencies: 45

## R=9 Theorem Chain

| id | statement | value |
|----|-----------|-------|
| N1 | In any exact 3x3 decomposition, Gamma*Sigma = 3I_9 forces rank(Gamma)=9. | rank_Gamma_is_9 |
| N2 | For R=9, Gamma is a 9x9 invertible matrix, so ker(Gamma) = {0}. | nullity_0 |
| N3 | Therefore Eta1 = 0, Eta2 = 0, and Delta = 0 as full matrices, not just after summation. | all_nuisance_zero |
| N4 | By Step 54 D1, Delta = 0 implies every nonzero term is dead-free and therefore supported on a unique summation index s*. | singleton_sum_index_support |
| N5 | By Step 54 D2, a dead-free term at s*=0 has (Sigma,Eta1,Eta2)=(v,v,0), at s*=1 has (v,-v,v), and at s*=2 has (v,0,-v). Hence Eta1=Eta2=0 implies v=0 and therefore Sigma=0 for that term. | no_nonzero_nuisance_free_term |
| N6 | So an exact R=9 decomposition would force every term to have zero Sigma row, contradicting rank(Sigma)=9 in Gamma*Sigma = 3I_9. | R9_impossible |
| N7 | Weaker corollary: if only Delta = 0 is imposed, then terms split by summation index and each summation channel must span the full 9-dimensional output-matrix space, requiring at least 9 terms per channel and therefore R >= 27. | deadfree_implies_R_at_least_27 |
| N8 | Hence any exact algorithm with R < 27 must use nonzero nuisance. Nuisance is structurally necessary, not an accidental by-product. | nuisance_is_load_bearing |

## Dead-Free Per-Channel Lower Bound

| summation index | active term shape | target dimension | minimum terms needed |
|-----------------|-------------------|------------------|----------------------|
| 0 | p_k q_k^T in M_3x3 | 9 | 9 |
| 1 | p_k q_k^T in M_3x3 | 9 | 9 |
| 2 | p_k q_k^T in M_3x3 | 9 | 9 |

## Nuisance Budget Table

| R | Gamma nullity | max nuisance rank | max dead rank | min dead dependencies | min total nuisance dependencies | all terms dead-free possible? |
|---|---------------|-------------------|---------------|-----------------------|--------------------------------|-------------------------------|
| 9 | 0 | 0 | 0 | 54 | 72 | False |
| 18 | 9 | 9 | 9 | 45 | 63 | False |
| 19 | 10 | 10 | 10 | 44 | 62 | False |
| 20 | 11 | 11 | 11 | 43 | 61 | False |
| 21 | 12 | 12 | 12 | 42 | 60 | False |
| 22 | 13 | 13 | 13 | 41 | 59 | False |
| 23 | 14 | 14 | 14 | 40 | 58 | False |
| 27 | 18 | 18 | 18 | 36 | 54 | True |

## R=18..23 Pressure Rows

| R | Gamma nullity | max nuisance rank | min dead dependencies | min total nuisance dependencies |
|---|---------------|-------------------|-----------------------|--------------------------------|
| 18 | 9 | 9 | 45 | 63 |
| 19 | 10 | 10 | 44 | 62 |
| 20 | 11 | 11 | 43 | 61 |
| 21 | 12 | 12 | 42 | 60 |
| 22 | 13 | 13 | 41 | 59 |
| 23 | 14 | 14 | 40 | 58 |

[INTERPRETATION]

The exact message is not that nuisance is unfortunate; it is that nuisance is mandatory. At R=9 the framework collapses immediately: Gamma has no nullspace, so all nuisance would have to vanish term-by-term, but Step 54 shows a nonzero dead-free term always carries anisotropy nuisance. More generally, if every term were dead-free then the three summation channels decouple and each channel would need 9 terms to span the full 3x3 output-matrix space, forcing R>=27. So every savings below 27 must come from channel-mixing terms with load-bearing nuisance. At R=18 the burden is already extreme: the full 72 nuisance columns must live in a 9-dimensional annihilator, and the 54 dead columns alone need at least 45 exact linear dependencies. That does not prove R=18 impossible, but it recasts the problem precisely: success below 27 requires deliberately engineered nuisance identities, not nuisance avoidance.