# Phase 16 Results

Phase 16 was stopped after the structural obstruction became clear: every warm-start route here remains in the AlphaTensor-derived k=4 identity basin, while an exact R=22 decomposition would require k=5 under the conservation law R + k = 27.

## 16a. Single-Term Removal Re-optimization

- Status: completed for all 23 removals.
- Best run: removed term t20.
- Best final loss: 1.1375932418815112e-05.
- Best final max-abs residual: 0.0007661152131220143.
- Second-best run: removed term t08 with max-abs residual 0.0009737509569905985.
- No run reached 1e-10 or 1e-15 max-abs residual.

## 16b. Pair-Structured Addition Search

- Status: stopped early after 260 completed trials out of the planned 400.
- Best observed run: config mask_011, removed terms t06,t12,t15, trial 8.
- Best final loss: 5.956340636230645e-05.
- Best final max-abs residual: 0.0012684318739524152.
- No completed trial reached 1e-10 or 1e-15 max-abs residual.

## 16c. Continuation from R=23 to R=22

- Status: stopped early after 78 completed continuation rows.
- Best observed run: removed term t02 at epsilon 0.01.
- Best final loss: 1.1867749090818628e-05.
- Best final max-abs residual: 0.00068993094228323.
- No completed continuation row reached 1e-10 or 1e-15 max-abs residual.

## Interpretation

The Phase 16 behavior is consistent with the identity-count obstruction.

- AlphaTensor starts with k=4.
- Phase 6 showed that all 22-term single removals still preserve k=4.
- Exact R=22 would require k=5 if R + k = 27 is the governing conservation law.
- The observed residual floor around 1e-3 to 1e-4 is therefore consistent with optimization inside the wrong identity stratum, not with a missing optimizer tweak.

Under that interpretation, Phase 16 has served its purpose: it confirms that warm-start continuation inside the AlphaTensor basin is not a viable route to an exact R=22 decomposition.