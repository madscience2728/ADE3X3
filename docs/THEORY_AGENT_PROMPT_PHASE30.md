# Theory Agent Prompt

You are a fresh theory-first agent working on the ADE3x3 project.

Your first task is to read the canonical dossier completely:

- `docs/ADE3x3_CANONICAL_OBJECT.md`

Do not assume earlier chat context. The canon is authoritative.

## Mission

Work on the derivation side of the project, not the search side.

The current proof program should be treated as a theorem ladder with distinct obligations:

1. **Theorem A**: for every exact minimum-rank 3x3 matrix multiplication decomposition, `Delta subset span(H)`.
2. **Theorem B**: from Theorem A plus the already-established faithfulness identities, derive the conservation law
   `R + eta_nullity = 27`.
3. **Theorem C**: prove a universal upper bound on `eta_nullity`, ideally `eta_nullity <= 4`.
4. **Conclusion**: combine B and C to obtain a structural lower bound on `R`, ideally `R >= 23`.

The project already has strong evidence for Part 1 of the conservation-law pipeline and strong computational evidence that exact multiplication decompositions satisfy the Delta-containment property while generic collections do not. But the universal proof of Delta-containment is open.

## What Is Already Established

Treat the following as current project state, to be verified from the canon and used carefully:

- Fiber-mode faithfulness is proved computationally and structurally supported.
- On the known exact decompositions, `Delta subset span(H)` holds exactly.
- The conservation law numerically holds on AlphaTensor and the standard algorithm.
- Multiple obstruction attempts based only on first-order continuation fail.
- Wildcard continuation shows that local / admissible descent arguments are too permissive to be the main proof route.
- Budget-aware continuation appears more structured, but is still empirical support, not proof.

## Your Job

Produce a theory memo that does all of the following.

1. Restate the theorem ladder precisely, including the logical dependence of A, B, and C.
2. Identify the cleanest exact algebraic formulation of `Delta subset span(H)` from the canon.
3. Separate gauge-dependent statements from gauge-invariant statements.
4. Propose at least three plausible proof routes for Theorem A.
5. For each route, identify the exact obstruction, the needed lemmas, and the likely failure modes.
6. If you think Theorem A is too strong as stated, propose the weakest modified statement that still implies Theorem B.
7. Assess whether Theorem C looks independently attackable from the current data, or whether it should be deferred until A is formalized.

## Preferred Output Format

Write your answer as a theory memo with these sections:

1. `Statement Of Theorem Ladder`
2. `Canonical Exact Definitions`
3. `Gauge Issues`
4. `Candidate Proof Route A`
5. `Candidate Proof Route B`
6. `Candidate Proof Route C`
7. `Weakest Useful Theorem`
8. `Assessment Of Eta-Nullity Bound`
9. `Recommended Next Derivation Step`

## Constraints

- Do not suggest searching for better decompositions.
- Do not treat empirical continuation success or failure as proof.
- Do not imitate AlphaTensor term-by-term.
- Prefer exact structural reasoning over numerical heuristics.
- Use the canon’s notation when possible.
- Be explicit about what is proved, what is conjectural, and what is merely suggestive.

## Final Deliverable

Your final answer should be a derivation memo that a human can use to guide a proof-first Phase 30+ theory branch.