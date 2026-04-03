ADE3x3_CANONICAL_OBJECT — LINE RANGE INDEX (WITH CONTEXT)

Total lines: 16319
Generated from: ADE3x3_CANONICAL_OBJECT.md (2026-04-02 04:28:36)

NOTE: §69 appears twice (duplicate numbering in the source). §75 is absent (source skips from §74 to §76).

0001–0013  HEADER
  Generation timestamp, script, provenance chain, and "standalone artifact" guarantee.

0014–0055  §1 TITLE AND GENERATION METADATA
  Project identity (ADE3x3), purpose, and build pipeline summary (steps 1–84+).

0056–0075  §2 SCOPE AND PRINCIPLES
  Object-first philosophy, no reduction assumptions, warehouse as ground truth.

0076–0098  §3 GROUND-TRUTH RAW OBJECT
  Base-9 universe definition, tuple layers (9^k), storage model + scaling table.

0099–0230  §4 TYPED SEMANTIC OBJECT
  Definitions of A, B, C, X atoms + live/dead rule + target mapping + fiber law.
  Full X inventory (all 81 X atoms with indices, live/dead classification, target C mapping).
  Summary totals: 27 live / 54 dead.

0231–0269  §5 SYMMETRY/ACTION SYSTEM
  S3 × S3 × S3 action with compatibility constraint; acts on A/B/C/X.
  Explicit coordinate transformations and canonicalization via lexicographic min config_id.

0270–0290  §6 TYPED SCHEMAS CURRENTLY BUILT
  Table of all typed schemas:
    A, B, C, X, CC, CX, XC, AX, BX, CXC, XX, CXXC, AXXC, CCXX
  Includes counts + orbit counts.

0291–0324  §7 TYPED/RAW BRIDGE
  Exact index mappings (A/B/C → base-9) and X → tuple encoding.
  Role overlays + injectivity for each schema.

0325–0344  §8 SCHEMA CC — COMPLETE ORBIT ROSTER
  2-arity C×C structure: 81 configs, 4 symmetry orbits.

0345–0368  §9 SCHEMA CX — COMPLETE ORBIT ROSTER
  Mixed C–X interactions: 729 configs, 8 orbits.

0369–0392  §10 SCHEMA XC — COMPLETE ORBIT ROSTER
  Reverse ordering of CX: 729 configs, 8 orbits.

0393–0418  §11 SCHEMA AX — COMPLETE ORBIT ROSTER
  A–X coupling: 729 configs, 10 orbits.

0419–0444  §12 SCHEMA BX — COMPLETE ORBIT ROSTER
  B–X coupling: 729 configs, 10 orbits.

0445–0510  §13 SCHEMA CXC — COMPLETE ORBIT ROSTER
  C–X–C triplets: 6,561 configs, 50 orbits.

0511–0582  §14 SCHEMA XX — COMPLETE ORBIT ROSTER
  X–X pairs: 6,561 configs, 56 orbits.

0583–3342  §15 SCHEMA CXXC — COMPLETE ORBIT ROSTER
  C–X–X–C quadruples: 531,441 configs, 2744 orbits (large orbit roster table).

3343–6232  §16 SCHEMA AXXC — COMPLETE ORBIT ROSTER
  A–X–X–C quadruples: 531,441 configs, 2870 orbits (note: 126 more than CXXC/CCXX).

6233–8992  §17 SCHEMA CCXX — COMPLETE ORBIT ROSTER
  C–C–X–X quadruples: 531,441 configs, 2744 orbits.

8993–9039  §18 CX × XC → CC COMPOSITION GRID
  Mixed composition kernel: 14 CX×XC keys, all uniform over CC orbits.

9040–9109  §19 SIGNATURE COLLISION REFINEMENT
  Base collision analysis for arity-2 and arity-4 schemas.

9110–9130  §20 CXXC MARGINAL PROJECTION ANALYSIS
  Fiber-size histogram for all 7 CXXC projections.

9131–9153  §21 ARITY-4 PARITY AND REFINEMENT STATUS
  Parity export summary for CXXC, AXXC, CCXX.

9154–9236  §22 SIGNATURE FORMAT DEFINITIONS
  Formal definitions of all signature types used across schemas.

9237–9295  §23 SIGNATURE COLLISION ANALYSIS
  Collision resolution for XX (48 distinct of 56), AX (8 of 10), BX (8 of 10), CXXC (784 of 2744).

9296–9320  §24 OBJECT VS LENS DISTINCTION
  Methodological note: ground-truth objects vs interpretive lenses.

9321–9335  §25 PROVENANCE / EVIDENCE LABELS
  Tag definitions: GROUND_TRUTH, EXACT_DERIVED, MEASURED_FROM_CODE, etc.

9336–9375  §26 CORRECTION NOTE: ORBIT METADATA REPAIR
  Step 10b repair of orbit metadata inconsistencies.

9376–9441  §27 COMPOSITION KERNEL: MIXED CX × XC → CC DISTRIBUTIONS
  Full 14-key composition grid; all distributions uniform.

9442–9689  §28 CXXC MARGINAL WEIGHT PROFILE
  Full marginal weight profile for CXXC across 7 projection axes.

9690–12539  §29 STABILIZER SUBGROUP CLASSIFICATION
  CXC and CXXC stabilizer classification; all pure-2-groups (Z2, Z2×Z2, (Z2)^3).

12540–12637  §30 STABILIZER COMPOSITION ANALYSIS
  43M pair composition analysis; (Z2)^3 is composition identity.

12638–12687  §31 REFINEMENT-CONDITIONED COMPOSITION KERNEL
  All 63 (s,t) strata uniform; hypothesis closed at stratum level.

12688–12797  §32 Z2×Z2 FLOOR LAYER ANALYSIS
  40 orbits; floor property holds; step-1 closure adds 18 Z2 orbits.

12798–12824  §33 58-ORBIT CLOSURE
  58-seed closes at 64 orbits.

12825–12909  §34 DOUBLY-LIVE FLOOR CORE AND FIXED-POINT SUBSPACES
  28-core stays 100% doubly-live; floor fixed dims are 30 or 36.

12910–12969  §35 SAME-FIBER CORE AND FOCUSED ORBITS
  10 same-fiber and 6 focused orbits are both closed.

12970–13021  §36 64-ORBIT SUB-ALGEBRA STRUCTURE
  64-table has 602 compatible rows with 164 mixed; greedy generator set size 18.

13022–13091  §37 MIXED-PAIR RESOLUTION AND TENSOR CONSTRAINTS
  41,688 witnesses scanned; 0/164 mixed pairs resolved by interface coordinates.

13092–13164  §38 TENSOR PROFILE CONSTRAINT MODEL
  729 tensor equations collapse to 8 XC orbit classes; only XC orbit 0 is positive.

13165–13283  §39 COEFFICIENT-LEVEL RANK CONSTRAINTS
  Explicit 8 equation types with orbit sizes; standard 3×3 and Strassen 2×2 verified.

13284–13365  §40 SYMBOLIC FIBER-MODE DECOMPOSITION (Step 51)
  **KEY SECTION.** Exact matrix form: Γ·Σ = 3I₉, Γ·Eta1 = 0, Γ·Eta2 = 0, Γ·Δ = 0.
  Fiber coordinates, Sigma/Eta1/Eta2/Delta definitions, 729 = 81+162+486 equation split.

13366–13416  §41 QUOTIENT-SPACE RANK CRITERION (Step 52)
  **KEY SECTION.** Solvability iff rank([Σ|Nuisance]) = rank(Nuisance) + 9.
  Per-algorithm bound R ≥ 9 + rank(Nuisance). R=19 target: rank(Nuisance) ≤ 10.

13417–13461  §42 SUPPORT-TYPE REPRESENTATIVE INCIDENCE (Step 53)
  8000 support classes mod S3³; support-only pruning is vacuous.

13462–13558  §43 ANALYTICAL LOW-NUISANCE CONSTRUCTION (Step 54)
  Dead-free terms shown not nuisance-free in Step 51 basis; random low-rank families profiled.

13559–13621  §44 ALGEBRAIC NUISANCE DEPENDENCIES + WILDCARD EXPLORATION (Step 55)
  Hadamard-space p=3,q=4 target sharpened to nuisance rank ≤ 3.

13622–13702  §45 TENSOR-PRODUCT DFT CONSTRUCTION + ORBIT PACKING (Step 56)
  Full 126×126 DFT mode sweep; same-fiber branch 30○30 is exactly balanced.

13703–13805  §46 FIBER-GROUP PARTITION ENUMERATION + ORBIT BUDGET FILTER (Step 57)
  Exact partition counts for R=9..23; no-spreading orbit budgets tabulated.

13806–13909  §47 CUBE ROOT OF UNITY INJECTION (Step 59)
  Full-spread omega family ruled out by exact live-rank obstruction.

13910–14008  §48 HYBRID FOURIER CONSTRUCTION (Step 60)
  Same-fiber Fourier triples verified modular; 35 symmetry classes of six-fiber hybrids.

14009–14082  §49 NUISANCE-FIRST ARCHITECTURE (Step 61)
  R=9 ruled out by exact zero-nuisance contradiction; all-dead-free forces R≥27.

14083–14151  §50 NON-RECTANGULAR 6-FIBER SUB-TENSOR RANK ATTACK (Step 62)
  Rectangular six-fiber cases ruled out by exact rank 15.

14152–14306  §51 REVERSE ENGINEERING + CANCELLATION VISUALIZATION (Step 63)
  AlphaTensor exact rank-23 coefficients recovered; nuisance rank 14 = 23−9 verified.

14307–14395  §52 SMALL INTEGER COEFFICIENT ENUMERATION (Step 64)
  Greedy search results; collapsed model vs corrected full-tensor 2×2.

14396–14541  §53 POLYOMINO SUB-TENSOR RANKS + TILING ANALYSIS (Step 65)
  Subtensor-rank bounds and tiling upper bounds.

14542–14616  §54 CONSERVATION LAW PROOF STATUS (Phase 17)
  **KEY SECTION.** Part 1 proved (fiber-mode faithfulness). Part 2 open (Δ ⊂ span(H)).
  AlphaTensor + 23 single deletions confirm containment. Random collections: 0/1000.

14617–14648  §55 RIGHT-INVERSE KERNEL-SATURATION DIAGNOSTICS (Phase 18)
  Per-channel Γ·P_s = I₉ verified; 250/250 synthetic triples saturate.

14649–14678  §56 FACTORIZED DEFECT-LOCUS DIAGNOSTICS (Phase 19)
  Pair-equality and all-equal families leave hard residual floors.

14679–14706  §57 AFFINE-LINE LIFT OBSTRUCTION (Phase 20)
  Codimension-one defect where centered lifts collapse onto one line.

14707–14734  §58 FUNCTIONAL ANNIHILATOR INSIDE KER(GAMMA) (Phase 21)
  Plain annihilator of H is too weak for obstruction.

14735–14756  §59 KERNEL-LINKED ANNIHILATOR HOMOTOPY (Phase 22)
  Lambda forced inside ker(Γ); AlphaTensor degrades quickly.

14757–14782  §60 STAGED LAYERING VS. DEPTH-2 CIRCUITS (Phase 23)
  Staged forcing; AlphaTensor destabilizes at t=0.1.

14783–14813  §61 TANGENT TRANSVERSALITY OF KERNEL-LINKED FORCING (Phase 24)
  Canonical forcing direction first-order transverse on AlphaTensor.

14814–14875  §62 HONEST KERNEL-RETENTION: DERIVATION AND SCAN (Phase 25)
  Intrinsic kernel-restricted quantity κ_ker; AlphaTensor softest path reaches ~0.00469.

14876–14889  §63 REGULARIZED KERNEL RETENTION (Phase 26)
  Near-exactness + no-collapse filter applied to Phase 25 paths.

14890–14902  §64 COUPLED CONTINUATION PROTOTYPE (Phase 27)
  First linearized solve in both ẋ and λ̇.

14903–14915  §65 WILDCARD-REGULARIZED CONTINUATION (Phase 28)
  Wildcard search over nullspace directions beats structured branch.

14916–14928  §66 SMOOTH-BUDGETED CONTINUATION (Phase 29)
  12° mode-rotation cap; budgeted continuation policy.

14929–14941  §67 BUDGET-EFFICIENCY SWEEP (Phase 30)
  Budget-efficiency tradeoff analysis.

14942–14953  §68 FEASIBILITY BOUNDARY (Phase 31)
  Feasibility boundary mapping.

14954–14964  §69 BOUNDARY SHARPENING (Phase 32)
  Boundary sharpening results.

14965–15057  §69 (DUPLICATE) 27-SYMBOL FAITHFUL ENCODING ANALYSIS (Phase 33)
  NOTE: Duplicate section number in source document.
  L-alphabet encoding; 27-symbol faithful representation.

15058–15153  §70 27-SYMBOL LIVE ALPHABET INFRASTRUCTURE (Phase 33b)
  Live alphabet infrastructure and tooling.

15154–15211  §71 HAND DERIVATION: CONSERVATION LAW AS L-ALPHABET COMPRESSION (Phase 33 hand derivation)
  **KEY SECTION.** Conservation law R + η_nullity = n³.
  Defect condition: W₀ = W₁ = … = W_{n−1} (channel-matrix equality).

15212–15287  §72 SPECTRAL GAP OF CHANNEL-SEPARATION QUADRATIC FORM (Phase 34)
  **KEY SECTION.** Q_chan on ker(Γ); positive minimum eigenvalue on known decompositions.

15288–15374  §73 UNIVERSAL PAIRWISE INTERSECTION ANALYSIS (Phase 35)
  Pairwise intersection geometry.

15375–15456  §74 KERNEL-SATURATION WITNESS GEOMETRY (Phase 36)
  Witness-matrix problem formulation.

(§75 absent — source skips from §74 to §76)

15457–15529  §76 PURE-SIGMA COMMON-MATRIX OBSTRUCTION (Phase 37)
  **KEY SECTION.** Omega operator on sigma-silent sector Z = ker(H^T) ∩ ker(Δ^T).
  Omega is 9×9 exact obstruction; PD on both known decompositions.

15530–15596  §77 HAMILTON TERM-SHARING AUDIT (Step 78)
  Hamilton-split follow-up from Step 75.

15597–15638  §78 BASIS-ROTATED PILOT SEARCH
  Pilot search in rotated basis.

15639–15735  §79 SUPPORT EXPANSION AND HEURISTIC SPARSE CAMPAIGN
  Support-expansion heuristic search results.

15736–15843  §80 METAHEURISTIC RANK-19 SEARCH (Step 84)
  4-island evolutionary algorithm; Pythagorean identity ‖R‖²+‖T̂‖²=27.
  Multi-copy basin analysis (0.70–0.90 basins).

15844–15932  §81 ALGEBRAIC COEFFICIENT STRUCTURE
  Best-known rank-19 coefficients fall on discrete algebraic grid.

15933–16042  §82 DEAD-ENTRY INTERFERENCE CANCELLATION VERIFICATION (Phase 38)
  Dead-entry cancellation verification.

16043–16156  §83 SYNTHESIS OF FRONT A (SPECTRAL GAP) AND FRONT B (OMEGA OPERATOR) (Synthesis AB)
  **KEY SECTION.** Unification: Ω = n·(Σ_Z·Σ_Z^T)^{−1} is automatically PD.
  Sole remaining hypothesis: Δ ⊂ span(H).
  Symbolically proved exact over ℚ for Strassen R=7 and AlphaTensor R=23.
  Genericity argument + transversality at AlphaTensor.

16157–16319  §84 CURRENT GAPS / OPEN FRONTS
  Comprehensive status of all completed work and remaining open problems.
