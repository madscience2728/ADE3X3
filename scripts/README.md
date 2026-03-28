# Scripts Directory

Utility scripts for running and managing the ADE3x3 pipeline.

## Available Scripts

### `run_pipeline.py`
Main orchestrator for running the step pipeline.

**Usage:**
```bash
# Run all steps
python scripts/run_pipeline.py

# Run steps 1 through 10
python scripts/run_pipeline.py 1-10

# Run step 5 only
python scripts/run_pipeline.py 5

# Run specific steps
python scripts/run_pipeline.py 1,3,5

# List available steps
python scripts/run_pipeline.py --list

# Stop on first error
python scripts/run_pipeline.py --stop-on-error
```

## Future Scripts (TODO)

- `validate.py` - Validate warehouse integrity
- `export_all.py` - Export all schemas and metadata
- `compare_versions.py` - Compare warehouse states
- `benchmark.py` - Performance benchmarking
