# ADE 3×3 RANK-19 DECOMPOSITION CHALLENGE

## THE PROBLEM

You must improve a rank-19 CP decomposition of the 3×3 matrix multiplication tensor T.

**Goal:** Lower the max-abs residual (currently 0.500) toward 0 by adjusting the 276 real-valued coefficients in the 19 rank-1 terms while keeping the same support pattern (which indices are nonzero).

**NOTE:** The JSON field `variable_count: 237` uses an internal threshold and does NOT match the actual number of coefficients. The true count is 276 = sum of all alpha_values, beta_values, gamma_values array lengths across 19 terms. Many entries are clamped near ±0.0001 — these are legitimate small values, not padding. Trust the arrays.

If you achieve max-abs = 0 exactly, you will have proven that 3×3 matrix multiplication requires only 19 scalar multiplications — an open problem in mathematics.

## THE TENSOR

T is a 9×9×9 tensor with exactly 27 nonzero entries, all equal to 1.0.

```
T[c, a, b] = 1.0  when c = 3*r + u, a = 3*r + s, b = 3*s + u  for r,s,u ∈ {0,1,2}
T[c, a, b] = 0.0  otherwise
```

Explicitly, the 27 live entries are:
```
(r,s,u) -> T[3r+u, 3r+s, 3s+u] = 1.0

(0,0,0) -> T[0,0,0]    (0,0,1) -> T[1,0,1]    (0,0,2) -> T[2,0,2]
(0,1,0) -> T[0,1,3]    (0,1,1) -> T[1,1,4]    (0,1,2) -> T[2,1,5]
(0,2,0) -> T[0,2,6]    (0,2,1) -> T[1,2,7]    (0,2,2) -> T[2,2,8]
(1,0,0) -> T[3,3,0]    (1,0,1) -> T[4,3,1]    (1,0,2) -> T[5,3,2]
(1,1,0) -> T[3,4,3]    (1,1,1) -> T[4,4,4]    (1,1,2) -> T[5,4,5]
(1,2,0) -> T[3,5,6]    (1,2,1) -> T[4,5,7]    (1,2,2) -> T[5,5,8]
(2,0,0) -> T[6,6,0]    (2,0,1) -> T[7,6,1]    (2,0,2) -> T[8,6,2]
(2,1,0) -> T[6,7,3]    (2,1,1) -> T[7,7,4]    (2,1,2) -> T[8,7,5]
(2,2,0) -> T[6,8,6]    (2,2,1) -> T[7,8,7]    (2,2,2) -> T[8,8,8]
```

## THE DECOMPOSITION FORMAT

A rank-19 CP decomposition is 19 rank-1 terms. Each term k has three sparse vectors:
- alpha_k (length 9, mostly zeros) with nonzeros at alpha_support positions
- beta_k (length 9, mostly zeros) with nonzeros at beta_support positions  
- gamma_k (length 9, mostly zeros) with nonzeros at gamma_support positions

**CRITICAL: How to build the vectors from JSON:**
```python
# For each term in data["terms"]:
alpha = np.zeros(9)  # start with all zeros
for idx, val in zip(term["alpha_support"], term["alpha_values"]):
    alpha[idx] = val   # set value at the support index
# Same for beta (from beta_support/beta_values) and gamma (from gamma_support/gamma_values)
```

The `alpha_support` array gives INDEX POSITIONS (0-8) into a length-9 vector.
The `alpha_values` array gives the VALUES at those positions (same length as alpha_support).
All other positions in the length-9 vector remain 0.0.

The candidate tensor is:
```
T_hat[c, a, b] = sum_{k=1}^{19} gamma_k[c] * alpha_k[a] * beta_k[b]
```

The residual is R = T_hat - T. We want max_{c,a,b} |R[c,a,b]| = 0.

**Verification script (use this exact code to check your answer):**
```python
import json, numpy as np

with open('candidate.json') as f:
    data = json.load(f)

T = np.zeros((9, 9, 9))
for r in range(3):
    for s in range(3):
        for u in range(3):
            T[r*3+u, r*3+s, s*3+u] = 1.0

candidate = np.zeros((9, 9, 9))
for term in data['terms']:
    a = np.zeros(9); b = np.zeros(9); g = np.zeros(9)
    for i, v in zip(term['alpha_support'], term['alpha_values']): a[i] = v
    for i, v in zip(term['beta_support'], term['beta_values']): b[i] = v
    for i, v in zip(term['gamma_support'], term['gamma_values']): g[i] = v
    candidate += np.einsum('c,a,b->cab', g, a, b)

R = candidate - T
print(f'max_abs = {np.max(np.abs(R)):.6f}')  # MUST be < 0.500 to beat current
print(f'fro     = {np.linalg.norm(R):.6f}')
```

## CONSTRAINTS

1. **Support is FIXED.** Do not add or remove nonzero positions. Only change the numerical values at existing support positions.
2. **All 19 terms must be present.** Do not delete terms.
3. **Return a valid JSON** in the exact same schema as the input.

## STRUCTURAL FACTS

- The S₃³ group (order 216) acts on the 27 live entries by permuting row/column/output indices.
- ||T||²_F = 27 (sum of 27 ones squared)
- At an ALS local minimum: ⟨R, T_hat⟩ ≈ 0, so ||R||² + ||T_hat||² ≈ 27
- The current candidate has ||R||² ≈ 8.0 and ||T_hat||² ≈ 19.0
- Live energy (residual on 27 live entries): 3.48
- Dead energy (residual on 702 dead entries): 4.53
- Dead leakage is the dominant bottleneck: the 19 rank-1 terms pollute dead entries

## CURRENT STATUS OF THE BEST CANDIDATE

```
max_abs_residual = 0.500  (the SINGLE worst entry out of 729)
fro_residual     = 2.829  (sqrt of sum of all 729 squared entries)
```

### Live entry residuals (27 entries, target = 1.0 for each):

Some live entries are nearly perfect, others are stuck at ~0.5:
```
NEAR-PERFECT (residual < 0.01):          STUCK (residual > 0.4):
  T[0,2,6] = 0.9997  (err -0.0003)        T[4,3,1] = 0.5008  (err -0.4992)
  T[1,1,4] = 0.9999  (err -0.0001)        T[5,3,2] = 0.5001  (err -0.4999)
  T[2,0,2] = 1.0000  (err +0.0000)        T[5,4,5] = 0.5004  (err -0.4996)
  T[2,2,8] = 1.0000  (err -0.0000)        T[6,7,3] = 0.5002  (err -0.4998)
  T[4,4,4] = 1.0000  (err -0.0000)        T[7,7,4] = 0.5000  (err -0.4999)
  T[5,5,8] = 1.0000  (err -0.0000)        T[8,7,5] = 0.5013  (err -0.4987)
  T[8,6,2] = 1.0000  (err -0.0000)        T[8,8,8] = 0.5137  (err -0.4863)
```

### Worst dead entries (target = 0.0 for each):
```
  T[5,3,1] = +0.4999    T[4,3,2] = +0.4991
  T[6,7,4] = -0.4999    T[1,6,1] = +0.4893
  T[7,7,3] = -0.4999    T[7,0,1] = +0.4893
  T[5,7,5] = -0.4991    T[7,8,8] = -0.4375
  T[8,4,5] = -0.4991    T[8,8,7] = -0.4375
```

### Pattern observation:
The ±0.5 residuals on live AND dead entries suggest a systematic half-integer structure. Many live entries are exactly 0.5 instead of 1.0, and nearby dead entries are ±0.5 instead of 0.0. This likely reflects a local minimum where certain rank-1 terms only "half-contribute" to their intended targets while leaking equally into neighbors.

## WHAT TO DO

Adjust the coefficient values (alpha_values, beta_values, gamma_values) for each of the 19 terms to minimize the maximum absolute residual across all 729 entries.

Key insight: you need to simultaneously:
- Push the ~0.5 live entries closer to 1.0
- Push the ~0.5 dead entries closer to 0.0
- Without disturbing the near-perfect entries

This is a nonlinear optimization over 276 real variables. The system is highly coupled — changing one term's coefficients affects many entries simultaneously.

## VERIFICATION

Your output will be verified using the exact Python script in the "DECOMPOSITION FORMAT" section above. Trust that script, not mental arithmetic. The current score to beat is **0.500**.

Common mistakes that produce 0.999 instead of 0.500:
- Treating alpha_values as a dense length-9 vector instead of sparse (it's NOT — you must scatter via alpha_support)
- Miscounting variables (there are 276 values across all support positions, not 237)
- Modifying values without actually running the reconstruction to verify

## OUTPUT FORMAT

Return a complete JSON object with the same schema as the input. The "fitness" and "fro_residual" and "verification" fields will be recomputed — just return your best coefficient values.

## THE CANDIDATE TO IMPROVE

```json
{"generation_found":184,"origin":"term_xover:(6,4,6)|(5,4,6)","fitness":0.4999805461292346,"fro_residual":2.829120330524366,"support_signature":"(6,4,6)","support_histogram":{"(3,4,7)":1,"(3,6,5)":1,"(4,3,4)":1,"(4,4,4)":1,"(4,6,7)":1,"(4,7,6)":1,"(5,2,6)":1,"(5,3,3)":1,"(5,3,4)":1,"(6,1,3)":1,"(6,3,2)":1,"(6,3,6)":1,"(6,4,1)":1,"(6,6,6)":1,"(6,7,5)":1,"(6,7,7)":1,"(7,2,6)":1,"(7,3,6)":1,"(7,7,6)":1},"variable_count":237,"newton_polished":false,"potential_exact_hit":false,"verification":{"coordinate_count":729,"max_abs_residual":0.4999805461292346,"fro_residual":2.829120330524366,"threshold":1e-08},"terms":[{"term_index":1,"support_signature":"(6,3,6)","alpha_support":[0,1,4,5,6,8],"beta_support":[2,3,5],"gamma_support":[0,1,2,3,5,8],"alpha_values":[-0.00010005244711284935,0.7535853825536961,-0.7074103735071365,-0.00010005244711284935,-0.00010005244711284935,-0.00010005244711284935],"beta_values":[-0.00010006620154365858,1.0319213904952411,0.058810196490055484],"gamma_values":[0.5276993665227101,-0.0001,0.4241548362023529,-0.7799886667791252,-0.027976583799677865,-0.02795423650868138]},{"term_index":2,"support_signature":"(6,7,7)","alpha_support":[0,1,5,6,7,8],"beta_support":[0,1,2,3,5,6,7],"gamma_support":[0,2,3,4,5,7,8],"alpha_values":[-0.00010004552134376011,-1.0331172981577688,-0.00010004552134376011,-0.00010004552134376011,0.031369616657794225,-0.00010004552134376011],"beta_values":[-0.0001,-0.0001,-0.0001,0.6485243684069223,-0.8048177049623684,-0.0001,-0.0001],"gamma_values":[-0.3635807852083402,0.9085285997409397,-0.33093084674926815,-0.00010006048444503109,-0.022922567866423924,0.011601313899058031,-0.022912240712819134]},{"term_index":3,"support_signature":"(4,7,6)","alpha_support":[0,1,2,5],"beta_support":[0,1,3,4,6,7,8],"gamma_support":[0,1,2,3,4,5],"alpha_values":[-0.00010000000362282764,-0.00010000000362282764,-0.7265338349247022,0.7310920712612131],"beta_values":[-0.0001,-0.0001,-0.0001,-0.0001,-0.02792766721070979,1.03032375516318,0.00012207381978352302],"gamma_values":[-0.020957479322440967,-0.8000209548628678,-0.0001,-0.3856465713525261,0.5226385058491213,-0.0001]},{"term_index":4,"support_signature":"(7,7,6)","alpha_support":[0,1,2,4,5,6,8],"beta_support":[0,2,3,4,6,7,8],"gamma_support":[0,3,5,6,7,8],"alpha_values":[-0.0001000000070754194,-0.0001000000070754194,-0.0001000000070754194,-0.0001000000070754194,0.0001000000070754194,-0.0001000000070754194,-1.030609775680568],"beta_values":[-0.0001,-0.0001,-0.0001,-0.0001,0.14745028854245376,-0.7208256685922851,0.721682269170853],"gamma_values":[-0.00010000000000000003,-0.00010000000000000003,-0.00010000000000000003,-0.5233533125891738,0.5148919193083908,-0.7232870532238355]},{"term_index":5,"support_signature":"(6,7,5)","alpha_support":[0,1,5,6,7,8],"beta_support":[0,1,3,4,6,7,8],"gamma_support":[0,5,6,7,8],"alpha_values":[-0.00010000000326568443,0.00010000000326568443,-0.00010000000326568443,0.00010000000326568443,0.0001544061694481043,-1.0306097673380734],"beta_values":[-0.00012187861523199072,-0.0001,-0.00010440259393632583,-0.00011006680478806231,0.8127932079727535,0.6156928906166758,-0.14981980517390944],"gamma_values":[-0.00011501689882349752,0.0001,-0.9554003964597835,-0.35298977578778956,-0.15736846500742488]},{"term_index":6,"support_signature":"(5,2,6)","alpha_support":[1,2,4,5,8],"beta_support":[6,7],"gamma_support":[0,1,2,3,4,7],"alpha_values":[0.00010000000210411818,0.021166694103664237,-0.00010000000210411818,1.0302864583851945,0.00010000000210411818],"beta_values":[0.8193839921449647,0.6249384919236186],"gamma_values":[0.008822721313438317,0.33679456994682616,-0.0001,0.9052910811835959,0.3589956278892266,-0.0001]},{"term_index":7,"support_signature":"(6,1,3)","alpha_support":[0,1,5,6,7,8],"beta_support":[4],"gamma_support":[1,4,7],"alpha_values":[-0.0001,1.0034015803840894,-0.0001,0.0001,0.013429032589391487,-0.0001],"beta_values":[1.003491460070077],"gamma_values":[0.9929928238243588,-0.14434479537177422,0.011177759882079722]},{"term_index":8,"support_signature":"(6,4,1)","alpha_support":[1,2,4,5,6,8],"beta_support":[2,3,4,5],"gamma_support":[4],"alpha_values":[-0.14435387994271262,-0.0001,-0.9930534622505912,-0.0001,-0.0001,-0.0001],"beta_values":[-0.0001,-0.0001,1.0034905238926781,-0.0001],"gamma_values":[-1.0034905388405022]},{"term_index":9,"support_signature":"(6,6,6)","alpha_support":[0,1,2,3,5,7],"beta_support":[0,3,4,6,7,8],"gamma_support":[0,1,2,3,5,8],"alpha_values":[0.00010000000625652533,-0.00010000000625652533,-0.00010000000625652533,0.00010000000625652533,1.0017206076224598,-0.00010000000625652533],"beta_values":[-0.0001,-0.0001,0.0001,-0.0001,-0.0001039624101746077,-1.0017206072190656],"gamma_values":[-0.0001,-0.00011270902259007516,-0.10147466571247672,-0.0001,-0.9965676471021592,-0.0001]},{"term_index":10,"support_signature":"(6,3,2)","alpha_support":[0,2,4,5,6,7],"beta_support":[0,4,8],"gamma_support":[2,4],"alpha_values":[-0.00010000000341221777,0.9965676367743935,0.00010000000341221777,-0.10147466444902557,0.00010000000341221777,-0.00010000000341221777],"beta_values":[-0.0001,-0.0001,-1.0017206108447327],"gamma_values":[-1.0017206158361445,-0.0001]},{"term_index":11,"support_signature":"(7,2,6)","alpha_support":[0,2,3,5,6,7,8],"beta_support":[0,1],"gamma_support":[0,1,3,4,6,7],"alpha_values":[0.6684231177296333,-0.00010028335520817517,0.20788665265730388,-0.00010028335520817517,-0.7145208876198285,-0.00010028335520817517,-0.000115595942135005],"beta_values":[-0.9897613465689333,-0.1446338269016513],"gamma_values":[-0.6790475014551258,-0.09961622866884783,-0.1702293219537616,-0.014410536697323085,0.7000803771896904,0.10109077432046104]},{"term_index":12,"support_signature":"(4,3,4)","alpha_support":[0,2,7,8],"beta_support":[5,6,7],"gamma_support":[0,2,4,7],"alpha_values":[-0.0001000000027884964,1.0002352415524718,-0.0001000000027884964,-0.00012186827784437281],"beta_values":[-0.0001,0.9998922303521176,-0.026193338406502373],"gamma_values":[0.999872569494203,0.0001,-0.02693321496721438,-0.0001]},{"term_index":13,"support_signature":"(3,4,7)","alpha_support":[4,7,8],"beta_support":[1,4,5,6],"gamma_support":[0,2,4,5,6,7,8],"alpha_values":[0.7068623845197973,-0.7074685794017197,0.00010000300987511206],"beta_values":[-0.00010000065583116482,0.0006697393740619739,1.0000828780352213,-0.00010000065583116482],"gamma_values":[0.02200589309498503,-0.0037532391404171027,-0.0001,0.7062728301432988,-0.0002368120691001996,-0.0004439859276673643,-0.707704960808715]},{"term_index":14,"support_signature":"(4,6,7)","alpha_support":[1,5,7,8],"beta_support":[0,2,3,4,6,8],"gamma_support":[0,1,3,4,6,7,8],"alpha_values":[0.009129338053324693,0.00010000182936709847,-1.0000052384254552,-0.00010000182936709847],"beta_values":[-0.0001,-0.0001,-0.7072949125823138,0.7069849420797818,-0.0001,-0.0001],"gamma_values":[0.002989299409422794,0.009459968681036409,0.0022765425295246085,-0.0013753543984891788,0.7071660144463197,-0.7070391025065678,0.0005254209495399016]},{"term_index":15,"support_signature":"(5,3,4)","alpha_support":[0,2,5,7,8],"beta_support":[1,2,3],"gamma_support":[0,2,4,6],"alpha_values":[-1.0000030371212418,-0.00010000076649563133,0.00010000076649563133,-0.00010000076649563133,0.00010000076649563133],"beta_values":[0.0001000002896383784,-1.0000030471214603,-0.0001000002896383784],"gamma_values":[-0.0001,0.9999939157606358,0.0042728053109651705,0.00012581258420139003]},{"term_index":16,"support_signature":"(5,3,3)","alpha_support":[0,1,5,6,7],"beta_support":[2,5,7],"gamma_support":[3,6,8],"alpha_values":[-0.00010000000328720986,0.00010000000328720986,0.00010000000328720986,0.9999999924910081,-0.00010000000328720986],"beta_values":[-1.0000000024910094,-0.0001,-0.0001],"gamma_values":[-0.0001,-0.0001,-1.0000000024910094]},{"term_index":17,"support_signature":"(7,3,6)","alpha_support":[0,2,3,5,6,7,8],"beta_support":[0,1,7],"gamma_support":[0,1,3,4,6,7],"alpha_values":[-0.4254631998082598,-0.0001,0.883008571791542,-0.0001,-0.1978473499641571,0.0001,-0.0001],"beta_values":[-0.9980768731591089,-0.06090345968167644,-0.00010000000000037934],"gamma_values":[0.45232735955317677,0.022903195051424758,-0.8746651823181595,-0.026293620632540825,0.17032488885097996,0.0009930682873976172]},{"term_index":18,"support_signature":"(3,6,5)","alpha_support":[0,5,6],"beta_support":[0,1,4,5,6,7],"gamma_support":[1,4,5,6,7],"alpha_values":[-0.7102534896024332,-0.000100002885224432,-0.7037443759181994],"beta_values":[-0.038463246081602766,0.9991179684196883,-0.00010001410836576166,-0.00010001410836576166,0.00010001410836576166,-0.00010001410836576166],"gamma_values":[-0.7101008494811822,-0.0033369639466855087,-0.0030393216087048073,0.030778046479792837,-0.7032107072181675]},{"term_index":19,"support_signature":"(4,4,4)","alpha_support":[0,3,7,8],"beta_support":[1,2,3,6],"gamma_support":[0,4,5,6],"alpha_values":[0.008559275619750298,-0.9997312065643194,-0.0001029141620777963,-0.0001029141620777963],"beta_values":[-0.7068449285464268,-0.7070403063111259,-0.0001,-0.0001],"gamma_values":[0.0032329328691064274,0.7060494294103054,0.707520090645475,0.02085282877210605]}]}
```
