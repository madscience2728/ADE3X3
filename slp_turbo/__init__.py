"""SLP Turbo — Concurrent GPU+CPU minimax optimizer.

Architecture:
  CandidatePool: Thread-safe in-RAM store of all candidates + fitnesses
  GPUEngine:     Continuous smooth Adam descent on 150K candidates (own thread)
  CPUEngine:     Continuous LP solves on best candidates (process pool)
  Orchestrator:  Coordinates engines, merges results, updates pool

Run:  python -m slp_turbo
      python -m slp_turbo --input my_seed.json --workers 48 --gpu-batch 150000
"""
