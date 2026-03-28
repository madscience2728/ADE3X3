# ADE3x3 Documentation

This directory contains the technical documentation for the ADE3x3 project.

## Available Documentation

### Core Documentation
- **[ADE3x3_CANONICAL_OBJECT.md](ADE3x3_CANONICAL_OBJECT.md)** - The definitive technical dossier
  - Complete object specification
  - Ground truth definitions
  - Warehouse structure
  - Symmetry system
  - Current results and measurements
  - Evidence taxonomy and provenance

### Step-Specific Documentation
- **[README_step13e.txt](README_step13e.txt)** - Step 13e notes
- **[README_step31.txt](README_step31.txt)** - Step 31 notes
- **[README_step32.txt](README_step32.txt)** - Step 32 notes

## Key Concepts

### Evidence Taxonomy
The canonical dossier uses a rigorous evidence labeling system:
- `[GROUND_TRUTH]` - Core definition, object structure
- `[EXACT_DERIVED]` - Results derived by exact computation
- `[MEASURED_FROM_CODE]` - Measured numeric outputs
- `[REPAIRED]` - Corrected after bug fixes
- `[INTERPRETATION]` - Analysis, not ground truth
- `[OPEN_FRONT]` - Known gaps or incomplete areas

### Object vs Lens Distinction
Critical principle: The **object** (raw warehouse, typed species, schemas, rules, bridges) must be kept distinct from **organizational lenses** (orbit metadata, signatures, canonicalization tables, caches).

A lens update does not change the underlying object.

## Reading Guidelines

1. **Start with the canonical dossier** for ground truth
2. **Cite sections precisely** when referencing facts
3. **Distinguish object facts from lens facts**
4. **Any reduced view must state what it omits**

## TODO: Future Documentation

- [ ] Architecture guide (system design overview)
- [ ] Pipeline documentation (step-by-step guide)
- [ ] API reference (when code is stabilized)
- [ ] Tutorial/quickstart guide
