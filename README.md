# ADE3x3 - Algebra Discovery Engine for Exact 3x3 Matrix Multiplication

> Systematic analysis and cataloging of the ambient bilinear universe for 3x3 matrix multiplication

## Overview

ADE3x3 is a research computation project that records and organizes exact data about 3x3 matrix multiplication. The project builds a comprehensive warehouse of all possible configurations, analyzes their symmetry properties, and catalogs their structural relationships.

**Key Principle:** The source of truth is the complete object stored in the warehouse — not a reduced model, not a surrogate, not a compressed summary.

## Project Structure

```
ADE3X3/
├── src/ade3x3/          # Main package
│   ├── core/            # Shared utilities and data structures
│   └── steps/           # Sequential computation pipeline (steps 1-32+)
│
├── experiments/         # Experimental/unmade implementations
├── outputs/exports/     # Generated data exports (CSV, MD)
├── tests/               # Test suite
├── docs/                # Technical documentation
└── scripts/             # Utility and orchestration scripts
```

## Core Concepts

### Atomic Species
- **A**: Left input basis atoms (9 atoms: A[r,s])
- **B**: Right input basis atoms (9 atoms: B[t,u])
- **C**: Output basis atoms (9 atoms: C[r,u])
- **X**: Ambient product atoms (81 atoms: X[r,s|t,u])

### Warehouse Layers
The project maintains a complete base-9 warehouse storing all k-tuples over the 9-symbol alphabet:

| Layer | Tuple Count |
|-------|-------------|
| 9^1   | 9           |
| 9^2   | 81          |
| 9^3   | 729         |
| 9^4   | 6,561       |
| 9^5   | 59,049      |
| 9^6   | 531,441     |
| 9^7   | 4,782,969   |
| 9^8   | 43,046,721  |
| 9^9   | 387,420,489 |

### Symmetry Group
- **Size**: 216 elements
- **Type**: S3 × S3 × S3 with compatibility constraint
- **Action**: Acts on A, B, C, and X atoms
- **Purpose**: All core schemas are canonicalized under group action

## Computational Pipeline

The project uses a **step-by-step pipeline** (steps 1-32+) where each step:
1. Imports previous steps as dependencies
2. Builds incrementally on warehouse state
3. Exports intermediate results
4. Can be run sequentially

### Key Pipeline Stages

**Phase 1: Foundation (Steps 1-9)**
- Permutation groups and actions
- Stabilizer computation
- Signature refinement
- Object database infrastructure

**Phase 2: Core Schemas (Steps 10-24)**
- Orbit metadata and canonicalization
- Schema exports (CC, CX, XC, AX, BX, XX, CXC)
- Typed/raw bridge construction
- Cross-schema alignment

**Phase 3: Higher-Arity (Steps 25+)**
- CXXC schema (first arity-4 typed schema)
- Extended schema families
- Composition analysis

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ADE3X3

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Running the Pipeline

```bash
# Run individual steps
python -m ade3x3.steps.step01_permutations
python -m ade3x3.steps.step02_stabilizers

# Or use the orchestrator (when available)
python scripts/run_pipeline.py
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=ade3x3
```

## Documentation

- **[Canonical Object Dossier](docs/ADE3x3_CANONICAL_OBJECT.md)** - Complete technical specification
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and structure (TODO)
- **[Pipeline Documentation](docs/PIPELINE.md)** - Step-by-step guide (TODO)

## Current Status

### Implemented
- ✅ Full raw base-9 warehouse (layers 1-9)
- ✅ Typed species and schema definitions
- ✅ Symmetry group actions and canonicalization
- ✅ Core schemas: A, B, C, X, CC, CX, XC, AX, BX, XX, CXC
- ✅ Orbit metadata and signature computation
- ✅ First arity-4 schema (CXXC)
- ✅ Cross-schema raw alignment
- ✅ Typed/raw bridge for all schemas

### Open Fronts
- ⬜ Higher-arity typed schema family expansion
- ⬜ Full orbit-complete signatures for XX, AX, BX
- ⬜ Unified raw-backed composition caches
- ⬜ Global closure/refinement in unified warehouse

## Key Results

### Orbit Counts (Repaired)
| Schema | Configs | Orbits | Avg Size | Stabilizer Range |
|--------|---------|--------|----------|------------------|
| XX     | 6,561   | 56     | 117.2    | 1-8              |
| CXC    | 6,561   | 50     | 131.2    | 1-8              |
| CX     | 729     | 8      | 91.1     | 1-8              |
| XC     | 729     | 8      | 91.1     | 1-8              |
| AX     | 729     | 10     | 72.9     | 2-8              |
| BX     | 729     | 10     | 72.9     | 2-8              |
| CC     | 81      | 4      | 20.2     | 6-24             |

### Live/Dead Classification
- **Live X atoms**: 27 (where s == t)
- **Dead X atoms**: 54
- **C fiber structure**: Each C[r,u] has exactly 3 live X atoms

## Contributing

This is a research project. Experimental implementations go in `experiments/` with clear status markers.

## License

[Specify license here]

## References

For detailed technical information, see the [Canonical Object Dossier](docs/ADE3x3_CANONICAL_OBJECT.md).
