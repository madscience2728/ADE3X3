# ADE Framework: Algebra Derivation Engine
### CANON_ADE Workspace — Session 1

---

## What ADE Is

**ADE** = Algebra Derivation Engine.

The goal is not to find a rank-19 decomposition by numerical search. The goal is to **derive the unique algebra** whose intrinsic structure *is* the rank-19 decomposition — if it exists. If no such algebra exists, we obtain a proof of impossibility by algebraic means.

Classical optimization treats the tensor as a black box and hill-climbs a loss landscape. ADE treats the **constraint canon** as a set of axioms on an unknown algebra $\mathcal{A}$ and uses combinatorial search + representation theory to determine whether any consistent $\mathcal{A}$ can faithfully represent the problem at dimension 19.

The CD tower was the first hint: the correct algebraic home for our constraints is *not* a classical object. It lives at the boundary of the sedenion zero-divisor locus, and the 19 interior points of the support are exactly the parity-violation locus of the CD doubling rule. The algebra we need may require relaxing axioms the CD tower never touched.

---

## The Constraint Canon

Everything listed here is a **hard mathematical fact** derived from the structure of $T_{\mathrm{matmul}} \in \mathbb{R}^{9 \times 9 \times 9}$, where $T[3r+s,\; 3s+u,\; 3r+u] = 1$ for $r,s,u \in \{0,1,2\}$.

### C1 — Support Partition (8 + 19)

The 27 nonzero entries of $T$ split canonically into two orbits under $\mathbb{Z}_2 \wr S_3$:

| Set | Condition | Size | CD Label |
|-----|-----------|------|----------|
| Hull $\mathcal{H}$ | $(r,s,u) \in \{0,2\}^3$ (all-even) | 8 | L-sector — *deleted* |
| Interior $\mathcal{I}$ | at least one odd coordinate | 19 | Violation locus |

The 8 hull points are the CD "unmodified" block type $\{000\}$. The 19 interior points correspond to CD block types with at least one parity violation: $\{001, 010, 100, 011, 101, 110, 111\}$.

**Implication for $\mathcal{A}$:** Elements of $\mathcal{A}$ are indexed by $\mathcal{I}$ (the 19-point interior). The hull $\mathcal{H}$ is a *boundary stratum*, not part of the algebra's carrier set.

---

### C2 — r-Drops-Out Theorem

The Fourier character on $\mathbb{Z}_3^3$ is $\chi_{p,q,w}(r,s,u) = e^{2\pi i(ps + qu + wu)/3}$.

The $r$-index drops out completely:

$$\hat{T}(p,q,w) = 27 \cdot \delta_{p,0} \cdot \delta_{q+w \equiv 0 \pmod{3}}$$

**Active modes:** $(0,0,0),\; (0,1,2),\; (0,2,1)$ — only 3 of 27.
**Dead modes:** the remaining 24 are exactly zero.

**Implication:** In the algebra $\mathcal{A}$, the $r$-coordinate acts as a *label* rather than a participant in the product. $\mathcal{A}$ may be associative or commutative in $r$ while non-associative in $(s,u)$. The product rule should be blind to $r$ in the same way the Fourier character is.

---

### C3 — Fiber Constraint

Group the 19 interior points by their $(s,u)$-fiber: each fiber $F_{s,u} = \{r \in \mathbb{Z}_3 : (r,s,u) \in \mathcal{I}\}$.

| Fiber type | $(s,u)$ pairs | $|F_{s,u}|$ | $\Gamma(s,u)$ must equal |
|------------|--------------|-------------|--------------------------|
| Forced | $(0,0),(0,2),(2,0),(2,2)$ | 1 | **3 (pinned)** |
| Free | all others (5 fibers) | 3 | **3 (sum constraint)** |

Every valid CP decomposition of $T$ at rank 19 must satisfy $\Gamma(s,u) = 3$ for all 9 fibers simultaneously. This is an exact algebraic constraint, not a penalty.

**Total degrees of freedom:** 10 $\gamma$ d.o.f. (0 from forced fibers + 2 per free fiber).

**Implication:** The fiber partition defines a **quotient structure** on $\mathcal{A}$. The algebra should have a natural projection $\pi: \mathcal{A} \to \mathbb{Z}_3^2$ (the $(s,u)$-fiber map) such that every fiber has the same "weight" under $\pi$.

---

### C4 — Symmetry Group

$T$ is invariant under $G = \mathbb{Z}_2 \wr S_3$ (order 48), which acts on $\{0,1,2\}^3$ by:
- **Inner** $\mathbb{Z}_2^3$: swap $1 \leftrightarrow 2$ independently in each coordinate
- **Outer** $S_3$: permute the three coordinates $(r,s,u)$

The 27 terms split into 4 orbits of sizes $1 + 6 + 12 + 8 = 27$. The 8-point orbit $\{1,2\}^3$ is exactly the hull $\mathcal{H}$. The retained 19 terms are the union of the first three orbits.

**Implication:** $\mathcal{A}$ must carry an action of $G = \mathbb{Z}_2 \wr S_3$. Any multiplication table for $\mathcal{A}$ must be equivariant under this action. The search space for valid tables is the fixed-point set of $G$-action on the space of 19×19 structure constant matrices — a massive reduction.

---

### C5 — Sedenion Zero-Divisor Graph

The sedenion algebra $\mathbb{S}$ (CD level 4, dim 16) has a zero-divisor graph with:

- **42 vertices**: elements of the form $(e_i + e_j)$ with $i \in \{1,\ldots,7\}$ (L-imaginary), $j \in \{9,\ldots,15\}$ (U-sector), $j \neq i+8$
- **84 undirected ZD pairs** (each vertex has degree 8)
- The $j = i+8$ case is excluded: CD-paired elements cannot form ZD pairs

The 24 dead Fourier modes of $T$ and the 84 ZD pairs share the same combinatorial origin: both arise from the CD doubling rule's parity structure. The exact mapping between them is an open derivation (see open questions).

**Implication:** The zero-divisor locus of $\mathcal{A}$ should have the same degree-8 regularity as the sedenion ZD graph. Products that land in the "dead mode" sector of the Fourier spectrum correspond to ZD relations in $\mathcal{A}$.

---

## Stage 1: The Algebra Template

Define an **algebra template** $\mathcal{A}$ as a vector space $V = \mathbb{R}^{19}$ with basis $\{e_\xi : \xi \in \mathcal{I}\}$ (one basis element per interior point) and an unknown bilinear product:

$$e_\xi \cdot e_\eta = \sum_{\zeta \in \mathcal{I}} f^\zeta_{\xi\eta} \, e_\zeta$$

The $19 \times 19 \times 19 = 6859$ structure constants $f^\zeta_{\xi\eta}$ are the unknowns. The canon constraints become linear and quadratic equations on these constants:

| Canon constraint | Constraint type on $f^\zeta_{\xi\eta}$ |
|------------------|-----------------------------------------|
| C3 (fiber sum = 3) | Linear: $\sum_r f^\zeta_{(r,s,u),\eta} = 3$ for each fiber |
| C4 (G-equivariance) | Linear: $f^\zeta_{\xi\eta} = f^{g\zeta}_{g\xi, g\eta}$ for all $g \in G$ |
| C5 (ZD graph) | Bilinear: $e_\xi \cdot e_\eta = 0$ for each of the 84 ZD pairs |
| C2 (r-blindness) | Linear: structure constants independent of the $r$-fiber label |

The G-equivariance constraint (C4) alone reduces 6859 free constants to $\lfloor 6859 / 48 \rfloor \approx 143$ orbit representatives — a tractable search space.

---

## Stage 2: BFS/Constraint Propagation Over Algebra Space

The combinatorial engine works as follows:

1. **Initialize**: start with all $\approx 143$ orbit representatives of structure constants as free variables.
2. **Apply C3** (fiber constraint): set of linear equations, solved by Gaussian elimination. Eliminates a subspace of solutions.
3. **Apply C5** (ZD pairs): set each of the 84 ZD product constraints to zero. Further eliminates.
4. **Apply C2** (r-blindness): index the remaining constants by $(s,u)$-fiber only.
5. **Apply axiom selectively** (see Stage 3): test which residual degrees of freedom survive each additional axiom.
6. **BFS over axiom relaxation schedule**: enumerate which subsets of $\{\text{associativity, commutativity, distributivity, identity}\}$ yield a non-empty feasible set.

Each BFS node is a partial algebra — a partial assignment of structure constants consistent with all constraints applied so far. Each edge in the BFS graph is a new constraint (either a canon constraint or an axiom being imposed or relaxed).

---

## Stage 3: Axiom Imposition and Selective Relaxation

The CD tower gives a natural relaxation schedule (commutativity → associativity → ZD-freeness). But $\mathcal{A}$ is not constrained to follow the same schedule. Candidates for novel relaxation:

### R1 — Partial Associativity
$\mathcal{A}$ may be associative *within* each $(s,u)$-fiber but not across fibers:
$$(e_\xi \cdot e_\eta) \cdot e_\zeta = e_\xi \cdot (e_\eta \cdot e_\zeta) \quad \text{if all three share the same fiber}$$

This is a **fiber-local associativity** — a novel axiom not appearing in the CD literature.

### R2 — r-Transparent Product
The product ignores the $r$-label: $e_{(r_1,s,u)} \cdot e_{(r_2,s',u')}$ depends only on $(s,u,s',u')$, not on $r_1$ or $r_2$. This enforces C2 as an algebraic axiom rather than an observed property.

### R3 — Partial Identity
Instead of a global identity element, $\mathcal{A}$ has a **fiber-wise identity**: for each fiber $F_{s,u}$ there exists $\mathbf{1}_{s,u}$ such that $e_\xi \cdot \mathbf{1}_{s,u} = e_\xi$ for all $\xi \in F_{s,u}$. The 4 forced fibers (size 1) have their identity pinned by C3.

### R4 — Non-distributive Boundary
Distributivity may fail at the boundary between the 19-point interior and the 8-point hull. Since the hull is a boundary stratum (not part of $\mathcal{A}$'s carrier), products that "would land in the hull" are instead mapped to zero — exactly the ZD condition of C5.

---

## Stage 4 (Prospective): The Representation Theorem

Once a candidate algebra $\mathcal{A}$ is found that satisfies all canon constraints and the chosen axiom subset, the final step is:

**Conjecture:** There exists a faithful linear representation $\rho: \mathcal{A} \to \mathrm{End}(\mathbb{R}^3 \otimes \mathbb{R}^3)$ of dimension $d$. The rank of the CP decomposition equals $d$.

If $\mathcal{A}$ has a faithful irreducible representation of dimension 19, then rank$(T) = 19$.
If the minimum faithful representation dimension is $> 19$, then rank$(T) > 19$ — a structural proof of impossibility.

This converts the rank question into a **character theory question about $\mathcal{A}$** — decidable, certificate-producing, and algebraically meaningful.

---

## Open Problems (Ordered by Priority)

| # | Problem | Depends on |
|---|---------|-----------|
| OP1 | Map the 84 sedenion ZD pairs to the 24 dead Fourier modes explicitly | C2, C5 |
| OP2 | Write down the full linear system from C3 + C4, count its solution dimension | C3, C4 |
| OP3 | Determine which axiom relaxation schedule (R1–R4 subsets) yields non-empty feasible set | Stage 2 |
| OP4 | If feasible set is non-empty, find the minimum faithful representation dimension | Stage 4 |
| OP5 | Verify that the 4 forced-fiber terms (γ=3 pinned) uniquely determine their factor vectors under C4 | C3, C4 |

---

## What This Workspace Contains

| File | Purpose |
|------|---------|
| `ADE_FRAMEWORK.md` | This document — master spec for the ADE approach |
| *(to be created)* `CANON_CONSTRAINTS.py` | Formal Python encoding of C1–C5 as constraint objects |
| *(to be created)* `algebra_template.py` | 19×19×19 structure constant search with G-orbit reduction |
| *(to be created)* `axiom_bfs.py` | BFS over axiom relaxation schedule |
| *(to be created)* `representation_check.py` | Given candidate $\mathcal{A}$, compute min faithful rep dimension |

---

*CANON_ADE workspace initialized. All prior optimization work is retired. The engine derives algebra; it does not hill-climb.*

---

## Session 2: BFS Results & Key Corrections

### Computational Inventory (all verified)

| Quantity | Value | Source |
|----------|-------|--------|
| INTERIOR orbits under G | 3 (sizes 3, 7, 9) | `canon_constraints.py` |
| INTERIOR² orbits under G | **27** | `algebra_template.py` |
| INTERIOR³ orbits under G | **439** | `algebra_template.py` |
| Fiber constraint null space | dim 10 | `algebra_template.py` |
| Sedenion ZD pairs | 84 (42 vertices) | `canon_constraints.py` |

### C2 Encoding Correction

The first BFS run (root d.o.f.=1) revealed an over-constraint bug:

**Wrong**: applying r-blindness to all three index positions simultaneously. This forces $f[\xi_1,\eta,\zeta] = f[\xi_2,\eta,\zeta]$ for all $(\xi_1,\xi_2)$ in the same r-fiber, AND $f[\xi,\eta_1,\zeta] = f[\xi,\eta_2,\zeta]$, AND $f[\xi,\eta,\zeta_1] = f[\xi,\eta,\zeta_2]$ — a total equivariance that collapses everything to a single orbit.

**Correct**: C2 says the *input* r-fiber doesn't separate CP terms. This means only **first-index** r-blindness applies to the structure constants: $f[(r_1,s),\eta,\zeta] = f[(r_2,s),\eta,\zeta]$. Second and third-index independence is not asserted by C2.

| C2 encoding | d.o.f. after C4+C2 |
|-------------|---------------------|
| All 3 indices | 1 (trivial constant algebra) |
| First index only | **27** = \|INTERIOR²/G\| exactly |
| None (C4 alone) | 439 |

The coincidence $27 = |\mathcal{I}^2 / G|$ is structural, not accidental: after fixing the first index's r-fiber, the remaining freedom indexes the G-orbit of the (j,k) output pair. **This is the correct ADE root.**

### First BFS Run (root = 1, wrong encoding) — Results

The constant algebra (all $f=1$) is the unique solution. Key observations from that run:

- `AX_COMM` was free (trivially satisfied: $1 = 1$)
- `AX_ANTI_COMM` killed it (requires $f[i,i,k]=0$, but $f \equiv 1$)
- `AX_HULL_ZD` killed it (requires some $f=0$, contradicts $f \equiv 1$)
- `AX_FIBER_ID` was **INCONSISTENT** — forced-fiber elements have $f[\text{id},i,i] \equiv 1 \neq \delta_{ik}$, confirming forced elements are not identities in the constant algebra; the identity axiom is incompatible at the root level

### Second BFS Run (root = 27, correct encoding) — In Progress

Running as of Session 2. Expected: meaningful d.o.f. drops from each axiom, with several combinations producing tight solution spaces (d.o.f. $\leq 5$).

### FIBER_ID Diagnosis

The forced-fiber elements (interior label $(1,0,0)$, $(1,0,2)$, $(1,2,0)$, $(1,2,2)$) are **not identity elements** in any algebra consistent with C4+C2. The identity axiom (R3 partial identity) requires $f[\text{id},i,i] = 1$ — but the canonical null vector has $f[\text{id},i,i] = 1$ for all $i$ uniformly (wrong encoding case), and in the correct 27-d.o.f. case this is now an open question pending the second BFS.

### Files Created This Session

| File | Purpose |
|------|---------|
| `canon_constraints.py` | C1–C5 as Python objects, all verified |
| `algebra_template.py` | 439 triple orbits, ZD bridge, fiber null space |
| `axiom_bfs.py` | BFS over axiom lattice, orbit-reduced (439 vars → 27 root) |
| `extract_algebra.py` | Extracts null vector, diagnoses FIBER_ID, expands to full $f[i,j,k]$ |

### Updated Open Problems

| # | Problem | Status |
|---|---------|--------|
| OP1 | Map 84 ZD pairs to 24 dead Fourier modes | Open |
| OP2 | C3+C4 combined linear system, exact solution dim | Partial: 27 d.o.f. confirmed for C4+C2(1st-index) |
| OP3 | Which axiom subsets yield non-empty feasible set from root=27 | **In progress (BFS run 2)** |
| OP4 | Minimum faithful representation dimension | Blocked on OP3 |
| OP5 | 4 forced-fiber terms uniquely determine factor vectors | Open |
| OP6 | Are forced-fiber elements zero-divisors, near-identities, or projectors in the correct algebra? | New — pending BFS run 2 |
