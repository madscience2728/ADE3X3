# Step 59: Cube Root of Unity Injection
Generated: 2026-03-28T15:32:00

[EXACT_DERIVED]

Step 59 splits the omega construction into two families. The full-spread family is impossible for representation-theoretic reasons in the live block; the same-fiber Fourier family is exact but recovers the standard 27-term algorithm rather than a low-rank breakthrough.

Full-spread live rank per output block: 1
Target live rank per output block: 9
Balanced 9-term full-spread failures: 702
Same-fiber j=1 nuisance rank: 1
Same-fiber j=1,2 pair nuisance rank: 4
Same-fiber j=0,1,2 triple nuisance rank: 8

## Task 2f: Full-Spread Obstruction

- F1: In the full-spread family, every frequency channel gives the same live contribution on a fixed output block: the all-ones vector across the 9 input fibers x 3 live summation coordinates.
- F2: The full-spread family has live-block column rank 1 per output block, whereas the target live block has fiber rank 9.
- F3: Therefore no sum of full-spread omega terms can realize the 3x3 multiplication tensor exactly, regardless of R.

## Task 3: Explicit 9-Term Full-Spread Attempt

| output fiber | channel j | label |
|--------------|-----------|-------|
| (0,0) | 0 | fiber=(0,0),j=0 |
| (0,1) | 1 | fiber=(0,1),j=1 |
| (0,2) | 2 | fiber=(0,2),j=2 |
| (1,0) | 0 | fiber=(1,0),j=0 |
| (1,1) | 1 | fiber=(1,1),j=1 |
| (1,2) | 2 | fiber=(1,2),j=2 |
| (2,0) | 0 | fiber=(2,0),j=0 |
| (2,1) | 1 | fiber=(2,1),j=1 |
| (2,2) | 2 | fiber=(2,2),j=2 |

Failure counts: live_target=0, live_off_target=216, dead=486, total=702

## Task 3e: Greedy Restricted-Family Augmentation

| R | selected term | live target fails | live off-target fails | dead fails | total fails | residual L2 |
|---|---------------|-------------------|-----------------------|------------|-------------|-------------|
| 1 | fiber=(0,0),j=0 | 27 | 24 | 54 | 105 | 5.1854497287 |
| 2 | fiber=(0,0),j=1 | 27 | 24 | 54 | 105 | 5.1747248988 |
| 3 | fiber=(0,1),j=0 | 27 | 48 | 108 | 183 | 5.1639777949 |
| 4 | fiber=(0,1),j=1 | 27 | 48 | 108 | 183 | 5.1532082779 |
| 5 | fiber=(0,2),j=0 | 27 | 72 | 162 | 261 | 5.1424162068 |
| 6 | fiber=(0,2),j=1 | 27 | 72 | 162 | 261 | 5.1316014394 |
| 7 | fiber=(1,0),j=0 | 27 | 96 | 216 | 339 | 5.1207638319 |
| 8 | fiber=(1,0),j=1 | 27 | 96 | 216 | 339 | 5.1099032389 |
| 9 | fiber=(1,0),j=2 | 27 | 96 | 162 | 285 | 5.0990195136 |
| 10 | fiber=(1,1),j=0 | 27 | 120 | 216 | 363 | 5.0881125075 |
| 11 | fiber=(1,1),j=1 | 27 | 120 | 216 | 363 | 5.0771820706 |
| 12 | fiber=(1,1),j=2 | 27 | 120 | 162 | 309 | 5.0662280512 |

At R=27 inside the full-spread family: total fails=243, live_off_target=216, dead=0

## Task 4: Same-Fiber Fourier Nuisance Profiles

| case | term count | sigma rank | eta rank | delta rank | nuisance rank | augmented rank | quotient gain |
|------|------------|------------|----------|------------|---------------|----------------|---------------|
| single_j1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| pair_j1_j2 | 2 | 1 | 2 | 4 | 4 | 4 | 0 |
| triple_j0_j1_j2 | 3 | 1 | 2 | 6 | 8 | 9 | 1 |

27-term same-fiber Fourier verification failures: 0

[INTERPRETATION]

The false R=3 and R=9 stories die for an exact reason: in the full-spread omega family, the live block cannot distinguish the 9 fibers. Adding more such terms can cancel dead-X, but it cannot create the missing fiber resolution, so the obstruction is permanent for that family. By contrast, the same-fiber Fourier bundle is exact and useful: three channels on one fiber kill dead-X by DFT orthogonality, but the bundle still has nuisance rank 2 before gamma annihilation. So omega injection is not a magic low-rank shortcut; in the viable same-fiber incarnation it is best understood as a Fourier re-expression of the standard algorithm, not a route below rank 27.