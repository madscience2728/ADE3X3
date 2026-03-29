# Wing It Compression Attempt

## Scope

- Target ranks attempted: 19 and 18
- Warm start source: public AlphaTensor rank-23 decomposition
- Search families: structured warm truncation, random warm truncation, border 23->19, cold random

## Best Results

- R=19: best max-abs residual = 0.0005268423997085019 via border23_to_19 (border_tail_t19_t12_t11_t04)
  quotient gain = 0, rank(H) = 18, rank(Nuisance) = 19

## Interpretation

- No exact hit at rank 19 or 18 in this pass.
- The exported best candidates are still useful as diagnostics for whether warm truncation or border-tail continuation is the less bad route.

