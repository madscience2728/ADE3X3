# REGRESSION WARNING — DO NOT USE NAIVE OPTIMIZATION

## What happened

On 2026-04-05, a `scipy.optimize.minimize` (L-BFGS-B) loss-based solver was added inside `operators.py` as `_build_factors_targeted_gate()`. This was **wrong** and contradicts the entire design philosophy of this engine.

## Why it's wrong

The axiom engine is a **combinatorial graph search with constructive algebraic operators**. Each BFS edge must be a deterministic algebraic construction — not "wiggle a vector of floats until a loss drops." Specifically:

- The engine does NOT compute residuals.
- The engine does NOT minimize loss functions.
- The engine does NOT do gradient descent, L-BFGS-B, Adam, or any other optimizer.
- The engine does NOT regress toward a target.

Every operator must be a **constructive** map: given algebraic structure X (kernels, irreps, fibers, parity blocks), produce algebraic structure Y by an explicit algebraic procedure.

## What the engine IS

- Axioms are graph nodes.
- Operators are constructive algebraic edges (e.g., decompose irreps, compute kernel complement, enumerate fiber partitions, verify parity).
- BFS discovers which axiom combinations are **reachable** at each rank via these constructions.
- If an axiom is unreachable, that's a real mathematical finding — not a convergence failure.

## The real problem with A5

The diagnostic showed that for R=13, `rank([Sigma | H | Delta]) = rank([H | Delta]) = R`. Sigma is trapped inside the nuisance space. This is a **dimension obstruction**, not an optimization gap. No amount of gradient descent fixes a rank constraint.

The correct next step is:
1. Map the dimension obstruction across all target ranks.
2. Find which ranks are geometrically feasible for A5.
3. For feasible ranks, design a **constructive** operator that builds factors satisfying the gate conditions by algebraic means (e.g., via the active basis structure, fiber constraints, or symmetry).

## Rule

If you see `scipy.optimize`, `minimize`, `loss`, `residual`, `gradient`, or `L-BFGS` inside `operators.py` or any axiom engine file — **delete it**. It does not belong here.


DO NOT TRY TO SOLVE MATMUL IN THE ENGINE! LET IT DROP OUT OF OF THE EGINE FOR FUCKS SAKE! THE ENGINE IS NOT A REGRESSOR! IT'S A COMBINATORIAL ALGEBRAIC CONSTRUCTION MACHINE! 
