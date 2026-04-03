# Wall-Breaker: Binding-Constraint Rotation

- [x] `binding.py` -- identify top-K binding entries at current best
- [x] `masked_lp.py` -- LP solve with one binding entry masked out
- [x] `coord_rotate.py` -- random orthogonal rotation of factor basis before/after LP
- [x] `strategy.py` -- strategy enum + per-worker assignment cycling
- [x] Wire into `cpu_engine.py` -- workers use assigned strategy each iteration
- [ ] Smoke test
