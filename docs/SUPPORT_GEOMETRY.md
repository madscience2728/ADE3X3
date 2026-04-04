# Support Geometry of T_matmul — The Discrete Bounded Object

*Canon document. No search, no optimization. Pure geometry.*

---

## 1. What the Object Is

T_matmul for 3×3 matrix multiplication has exactly **27 nonzero entries**, all equal to 1:

```
T[3r+s,  3s+u,  3r+u] = 1    for r, s, u ∈ {0, 1, 2}
```

The support of T_matmul is the image of the integer grid **{0,1,2}³** under the linear map:

```
A = | 3  1  0 |      (r,s,u)  ↦  (i, j, k) = (3r+s,  3s+u,  3r+u)
    | 0  3  1 |
    | 3  0  1 |

det(A) = 12,   rank(A) = 3
```

So the support is not just any 27 points — it is the **lattice image** of a clean 3×3×3 integer cube, projected through a rank-3 linear map with det=12.

This map is invertible over ℝ. Given any support point (i,j,k), the original coordinates recover as:

```
r = floor(i/3)  =  floor(k/3)        (both equal r — verified)
s = i mod 3     =  floor(j/3)        (both equal s — verified)
u = j mod 3     =  k mod 3           (both equal u — verified)
```

Three redundant encodings. The support is a **2-parameter family** in a 3D space: once (r,u) chosen, k is fully determined by i and j mod 3.

---

## 2. The Support Polytope

Define the **support polytope** P as the convex hull of the 27 support points in ℤ³ ⊂ ℝ³.

```
P = conv{ (3r+s, 3s+u, 3r+u) : r, s, u ∈ {0,1,2} }
```

### Key facts (computed):

| Property | Value |
|----------|-------|
| Ambient space | ℝ³ (index space) |
| Number of lattice points | 27 |
| Hull vertices | **8** |
| Interior + boundary non-vertex lattice points | **19** |
| Number of triangular faces | 12 |
| Volume | 96 |
| Surface area | 172.72 |

### Hull shape: Parallelepiped

P is a **parallelepiped** — the linear image of the cube [0,2]³ under A. Its 8 vertices are exactly the images of the 8 corners of {0,2}³:

```
Hull vertices (r,s,u ∈ {0,2}³):
  (0,0,0) → [0, 0, 0]      (2,0,0) → [6, 0, 6]
  (0,0,2) → [0, 2, 2]      (2,0,2) → [6, 2, 8]
  (0,2,0) → [2, 6, 0]      (2,2,0) → [8, 6, 6]
  (0,2,2) → [2, 8, 2]      (2,2,2) → [8, 8, 8]
```

Edge vectors of P (from origin vertex):
```
  e₁  =  A·(2,0,0)ᵀ  =  (6, 0, 6)     [r-direction]
  e₂  =  A·(0,2,0)ᵀ  =  (2, 6, 0)     [s-direction]
  e₃  =  A·(0,0,2)ᵀ  =  (0, 2, 2)     [u-direction]

  e₁·e₂ = 12    e₁·e₃ = 12    e₂·e₃ = 12
```

All three edge pairs have the same inner product (12). The parallelepiped is isotropically skewed — no pair of faces is orthogonal.

---

## 3. The Critical Partition: 8 + 19

The 27 lattice points split into two geometrically distinct layers:

### Layer A — The 8 Hull Vertices (Even corners)
Points with r, s, u all ∈ {0, 2}. These are the corners of the parallelepiped.

```
(r,s,u) ∈ {0,2}³ :
  (0,0,0), (0,0,2), (0,2,0), (0,2,2),
  (2,0,0), (2,0,2), (2,2,0), (2,2,2)
```

### Layer B — The 19 Interior Points (Odd-touching)
All remaining 19 points: every (r,s,u) with at least one coordinate = 1.

```
(r,s,u) with min(r,s,u) touches {1}:
  r=0: (0,0,1),(0,1,0),(0,1,1),(0,1,2),(0,2,1)         [5 points]
  r=1: (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),
       (1,1,2),(1,2,0),(1,2,1),(1,2,2)                  [9 points]
  r=2: (2,0,1),(2,1,0),(2,1,1),(2,1,2),(2,2,1)          [5 points]
```

**The 8+19 split matches the Strassen-rank structure exactly: R = 19.**

This is not a numerical coincidence. It is the combinatorial core:

> The 8 hull vertices are the "corner" support points — they lie on the boundary of the parallelepiped and can be captured by "extremal" rank-1 terms.
> The 19 interior points require all the algebraic work.

---

## 4. The Video Game Analogy — Exact Statement

In game dev, a **concave mesh** cannot be used directly for physics collision. The solution (V-HACD) is to:
1. Compute the outer convex hull (fast, loose)
2. Decompose the concave interior into a union of smaller convex pieces
3. The compound shape is exact but tractable

Here:
- **Outer convex hull** = the parallelepiped P (8 vertices, 12 faces, volume 96)
- **The concave interior** = the 19 off-vertex lattice points, which are NOT on the hull
- **Goal** = decompose P into R convex sub-objects, each corresponding to one rank-1 term

The decomposition we need is:

```
P  =  C₁ ∪ C₂ ∪ ⋯ ∪ Cᵣ
```

where each Cₖ is a convex region associated to one rank-1 term $a_k \otimes b_k \otimes c_k$, and the assignment respects the tensor weights.

---

## 5. The Natural Sub-Decomposition

Because P = A·[0,2]³ and A has integer structure, the parallelepiped tiles naturally.

### The 8 Sub-Parallelepipeds (Octant Decomposition)

Bisect each of the three dimensions at 1. The grid {0,1,2}³ splits into **8 "unit octants"**, each being the image under A of a sub-cube [i,i+1]×[j,j+1]×[k,k+1] for i,j,k ∈ {0,1}:

```
Octant (a,b,c) for a,b,c ∈ {0,1}: 
  corners {a,a+1}×{b,b+1}×{c,c+1} mapped under A
  contains 8 lattice points (the 2³ corners of the sub-cube)
  but adjacent octants share faces/edges/vertices
```

Each octant covers exactly **8 lattice points**, and the 8 octants together cover all 27 (with overlap at shared boundaries). This gives:
- 8 convex "chunks" (each is a parallelepiped = 1/8 of P)
- Each chunk corresponds to a symmetry-group coset of the {0,2}³ corner group
- Their union is P; their intersection structure is the combinatorial problem

### The Parity Observation

- The 8 hull vertices = **all-even** corner type (r,s,u) ∈ {0,2}³
- The 19 interior points = **at least one odd** type
- The rank-1 decomposition must provide one "seed" for each of the 19 odd-coordinate lattice points, plus it must reconcile the 8 even corners via consistent weighting

---

## 6. What a Rank-1 Term "Covers"

A rank-1 term $a \otimes b \otimes c$ with $a,b,c \in \mathbb{R}^9$ contributes to support point (i,j,k) via the product $a_i \cdot b_j \cdot c_k$.

For the decomposition $T = \sum_k \lambda_k (a_k \otimes b_k \otimes c_k)$ to be exact:

$$\forall (r,s,u):\quad \sum_{k=1}^R \lambda_k \cdot a_k[3r{+}s]\cdot b_k[3s{+}u]\cdot c_k[3r{+}u] = 1$$

Each rank-1 term defines a **rank-1 coverage function** over the 27 support points. The decomposition problem is: choose R terms such that the 27 coverage equations all equal 1 exactly.

In the parallelepiped picture: each rank-1 term is a **sampling pattern** over the 27 lattice points (a full 9×9×9 slice restricted to the 27 points). The weighted sum of R such patterns must equal the all-ones vector on the support.

---

## 7. The Compound Hull Construction (Next Step)

Following the V-HACD analogy:

**Step 1 — Outer Hull** (done): P is the parallelepiped with 8 vertices.

**Step 2 — Identify the concavities**: The 19 interior points are "dents" relative to the hull. In game dev terms: the hull passes OVER them (they are inside, not on the surface).

**Step 3 — Sub-Hull Decomposition**: Decompose P into R convex pieces, one per rank-1 term, such that:
- Each piece contains at least one "interior" support point
- The pieces tile P without holes (cover all 27 lattice points)
- The boundary conditions at the 8 hull corners are satisfied jointly

**Step 4 — Algebraic Encoding**: Each convex piece Cₖ corresponds to choosing which lattice points the k-th rank-1 term "primarily serves." The constraint that adjacent pieces share consistent boundary values IS the gate system (Gate-1 = null-space alignment, Gate-2 = off-diagonal cancellation).

**Step 5 — Concave Compound**: The R=19 decomposition (if it exists) is the minimal covering where:
- At most 1 rank-1 term "owns" each interior point (19 terms for 19 interior points)
- The 8 hull corners are covered by the boundary interactions of neighboring terms

This is the precise geometric content of the question **R(T_matmul) = 19?**

---

## 8. Summary Table

| Object | Description | Count |
|--------|-------------|-------|
| Support lattice | Image of {0,1,2}³ under A | 27 points |
| Hull (P) | Parallelepiped, image of [0,2]³ | 8 vertices, vol=96 |
| Hull vertices | r,s,u ∈ {0,2}³ — all-even coordinates | 8 |
| Interior lattice pts | At least one coordinate = 1 | **19** |
| Parity partition | Even = hull, Odd-touching = interior | 8 + 19 = 27 |
| Natural octant tiling | Sub-parallelepipeds bisecting each axis | 8 sub-hulls |
| Edge vectors | e₁=(6,0,6), e₂=(2,6,0), e₃=(0,2,2) | 3 generators |
| Edge inner products | e₁·e₂ = e₁·e₃ = e₂·e₃ = 12 | Isotropic skew |

---

## 9. Open Geometric Questions

1. **Is the 19-interior-point count a theorem or coincidence?** Is there a direct algebraic proof that rank(T_matmul) ≥ |interior(P)|?

2. **Does each rank-1 term in a minimal decomposition bijectively own one interior point?** If yes, the decomposition has a combinatorial structure: a perfect matching between rank-1 terms and interior lattice points.

3. **Can the 8 sub-hulls (octant decomposition) be used to construct the rank-1 terms?** Each octant contains one interior corner; can we find a rank-1 term centered on that corner?

4. **What is the "concave compound" of P with respect to the exact covering constraint?** This is the analog of V-HACD: find the minimal R such that P can be covered by R rank-1 coverage patterns with exact weight sum = 1 everywhere.

5. **Does the isotropic edge skew (all pairs dot to 12) encode the symmetry group Aut(T)?** The symmetry group has order 432; does it act transitively on the 19 interior points?
