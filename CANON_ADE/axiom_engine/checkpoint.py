"""
checkpoint.py — Checkpoint save/resume for overnight runs.
Stores results, worker state, and progress atomically.
"""
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import threading

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class CheckpointManager:
    def __init__(self, run_name=None):
        self.run_name = run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.dir = CHECKPOINT_DIR / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._results = []
        self._metadata = {}
        self._save_counter = 0
    
    def save_result(self, result):
        """Thread-safe append of a single result."""
        with self._lock:
            self._results.append(result)
            self._save_counter += 1
            if self._save_counter % 50 == 0:
                self._flush()
    
    def save_results_batch(self, results):
        """Thread-safe append of multiple results."""
        with self._lock:
            self._results.extend(results)
            self._save_counter += len(results)
            if self._save_counter % 50 == 0:
                self._flush()
    
    def set_metadata(self, key, value):
        with self._lock:
            self._metadata[key] = value
    
    def _flush(self):
        """Write current state to disk."""
        state = {
            'run_name': self.run_name,
            'timestamp': datetime.now().isoformat(),
            'n_results': len(self._results),
            'metadata': self._metadata,
        }
        state_path = self.dir / "state.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, cls=NumpyEncoder, indent=2)
        
        # Save results in chunks to avoid giant files
        chunk_size = 1000
        for i in range(0, len(self._results), chunk_size):
            chunk = self._results[i:i+chunk_size]
            chunk_path = self.dir / f"results_{i//chunk_size:04d}.json"
            with open(chunk_path, 'w') as f:
                json.dump(chunk, f, cls=NumpyEncoder)
    
    def flush(self):
        with self._lock:
            self._flush()
    
    def finalize(self):
        """Final flush + write summary."""
        with self._lock:
            self._flush()
            summary = self._compute_summary()
            with open(self.dir / "summary.json", 'w') as f:
                json.dump(summary, f, cls=NumpyEncoder, indent=2)
        return summary
    
    def _compute_summary(self):
        if not self._results:
            return {'n_results': 0}
        
        # Aggregate stats across all results
        keys = set()
        for r in self._results:
            keys.update(r.keys())
        
        summary = {
            'n_results': len(self._results),
            'ranks_tested': sorted(set(r.get('R', 0) for r in self._results)),
        }
        
        # Per-axiom pass rates
        for key in sorted(keys):
            if key.endswith('_pass') or key.endswith('_ok'):
                vals = [r[key] for r in self._results if key in r]
                if vals:
                    summary[f'{key}_rate'] = sum(vals) / len(vals)
        
        return summary
    
    @classmethod
    def resume(cls, run_name):
        """Resume from a previous checkpoint."""
        d = CHECKPOINT_DIR / run_name
        if not d.exists():
            raise FileNotFoundError(f"No checkpoint found: {d}")
        
        mgr = cls(run_name)
        
        # Load existing results
        for p in sorted(d.glob("results_*.json")):
            with open(p) as f:
                mgr._results.extend(json.load(f))
        
        # Load metadata
        state_path = d / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            mgr._metadata = state.get('metadata', {})
        
        return mgr
    
    @property
    def n_completed(self):
        with self._lock:
            return len(self._results)
    
    def get_completed_ids(self):
        """Return set of (rank, seed) pairs already completed."""
        with self._lock:
            return {(r.get('R'), r.get('seed_id')) for r in self._results}
