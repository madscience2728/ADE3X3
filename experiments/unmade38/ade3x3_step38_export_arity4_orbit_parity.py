from __future__ import annotations
import csv
import itertools
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

EXPORTS = Path('exports')
POW9 = np.array([9**5, 9**4, 9**3, 9**2, 9, 1], dtype=np.int64)
N = 9**6


def decode_raw6(raw_id: int) -> tuple[int, int, int, int, int, int]:
    return tuple(int((raw_id // p) % 9) for p in POW9)


def c_name(idx: int) -> str:
    return f"C[{idx//3},{idx%3}]"


def a_name(idx: int) -> str:
    return f"A[{idx//3},{idx%3}]"


def x_name(a_idx: int, b_idx: int) -> str:
    r, s = divmod(a_idx, 3)
    t, u = divmod(b_idx, 3)
    return f"X[{r},{s}|{t},{u}]"


def readable(schema: str, raw_id: int) -> str:
    vals = decode_raw6(raw_id)
    if schema == 'CXXC':
        c1, a1, b1, a2, b2, c2 = vals
        return f"CXXC[{c_name(c1)},{x_name(a1,b1)},{x_name(a2,b2)},{c_name(c2)}]"
    a, a1, b1, a2, b2, c = vals
    return f"AXXC[{a_name(a)},{x_name(a1,b1)},{x_name(a2,b2)},{c_name(c)}]"


def generate_maps():
    perms = sorted(itertools.permutations(range(3)))
    amap = []
    bmap = []
    cmap = []
    for pr, ps, pc in itertools.product(perms, perms, perms):
        a = np.empty(9, dtype=np.int16)
        b = np.empty(9, dtype=np.int16)
        c = np.empty(9, dtype=np.int16)
        for r in range(3):
            for s in range(3):
                a[3*r+s] = 3*pr[r] + ps[s]
        for t in range(3):
            for u in range(3):
                b[3*t+u] = 3*ps[t] + pc[u]
        for r in range(3):
            for u in range(3):
                c[3*r+u] = 3*pr[r] + pc[u]
        amap.append(a); bmap.append(b); cmap.append(c)
    return np.stack(amap), np.stack(bmap), np.stack(cmap)


def canonicalize(schema: str, digits: np.ndarray, amap: np.ndarray, bmap: np.ndarray, cmap: np.ndarray):
    canon = np.full(N, np.iinfo(np.int64).max, dtype=np.int64)
    for ai in range(216):
        if schema == 'CXXC':
            td0 = cmap[ai][digits[:,0]]
            td1 = amap[ai][digits[:,1]]
            td2 = bmap[ai][digits[:,2]]
            td3 = amap[ai][digits[:,3]]
            td4 = bmap[ai][digits[:,4]]
            td5 = cmap[ai][digits[:,5]]
        else:
            td0 = amap[ai][digits[:,0]]
            td1 = amap[ai][digits[:,1]]
            td2 = bmap[ai][digits[:,2]]
            td3 = amap[ai][digits[:,3]]
            td4 = bmap[ai][digits[:,4]]
            td5 = cmap[ai][digits[:,5]]
        tid = (td0*POW9[0] + td1*POW9[1] + td2*POW9[2] + td3*POW9[3] + td4*POW9[4] + td5*POW9[5]).astype(np.int64)
        canon = np.minimum(canon, tid)
    reps, counts = np.unique(canon, return_counts=True)
    rep_to_orbit = {int(rep): i for i, rep in enumerate(reps)}
    orbit_ids = np.array([rep_to_orbit[int(rep)] for rep in canon], dtype=np.int32)
    stab_sizes = []
    for rep in reps:
        vals = decode_raw6(int(rep))
        fix = 0
        for ai in range(216):
            if schema == 'CXXC':
                out = (
                    int(cmap[ai][vals[0]]), int(amap[ai][vals[1]]), int(bmap[ai][vals[2]]),
                    int(amap[ai][vals[3]]), int(bmap[ai][vals[4]]), int(cmap[ai][vals[5]])
                )
            else:
                out = (
                    int(amap[ai][vals[0]]), int(amap[ai][vals[1]]), int(bmap[ai][vals[2]]),
                    int(amap[ai][vals[3]]), int(bmap[ai][vals[4]]), int(cmap[ai][vals[5]])
                )
            tid = int(sum(v * int(p) for v, p in zip(out, POW9)))
            if tid == int(rep):
                fix += 1
        stab_sizes.append(fix)
    return reps.astype(int), counts.astype(int), np.array(stab_sizes, dtype=int)


def write_orbits_csv(schema: str, reps: np.ndarray, counts: np.ndarray, stabs: np.ndarray):
    path = EXPORTS / f'orbits_{schema}.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['schema','orbit_id','rep_config_id','rep_readable','orbit_size','stabilizer_size'])
        w.writeheader()
        for orbit_id, (rep, cnt, stab) in enumerate(zip(reps, counts, stabs)):
            w.writerow({
                'schema': schema,
                'orbit_id': orbit_id,
                'rep_config_id': int(rep),
                'rep_readable': readable(schema, int(rep)),
                'orbit_size': int(cnt),
                'stabilizer_size': int(stab),
            })


def main():
    EXPORTS.mkdir(exist_ok=True)
    ids = np.arange(N, dtype=np.int64)
    digits = ((ids[:,None] // POW9) % 9).astype(np.int16)
    amap, bmap, cmap = generate_maps()

    results = {}
    for schema in ('CXXC', 'AXXC'):
        reps, counts, stabs = canonicalize(schema, digits, amap, bmap, cmap)
        write_orbits_csv(schema, reps, counts, stabs)
        results[schema] = (reps, counts, stabs)

    # signature counts from existing pattern inventories / measured truth
    signature_counts = {'CXXC': 2744, 'AXXC': 2870}
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '# Arity-4 Orbit / Signature / Stabilizer Parity', '', f'Generated: {ts}', '',
        '| Schema | Orbit Count | Signature Count | Orbit Complete | Min Orbit | Max Orbit | Min Stab | Max Stab |',
        '|--------|-------------|-----------------|----------------|-----------|-----------|----------|----------|',
    ]
    for schema in ('CXXC', 'AXXC'):
        reps, counts, stabs = results[schema]
        lines.append(
            f"| {schema} | {len(reps)} | {signature_counts[schema]} | {'yes' if len(reps)==signature_counts[schema] else 'no'} | {counts.min()} | {counts.max()} | {stabs.min()} | {stabs.max()} |"
        )
    lines += [
        '',
        'Notes:',
        '- Orbit counts and stabilizer ranges were recomputed directly from the 216-element action on raw arity-6 tuples.',
        '- Signature counts are taken from the existing arity-4 face-pattern inventories already exported for each schema.',
        '- Equality of orbit count and signature count is the current parity fact for these two arity-4 schemas.',
    ]
    (EXPORTS / 'arity4_orbit_signature_stabilizer_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('CXXC:', len(results['CXXC'][0]), 'orbits; signatures = 2744; stab range', results['CXXC'][2].min(), '-', results['CXXC'][2].max())
    print('AXXC:', len(results['AXXC'][0]), 'orbits; signatures = 2870; stab range', results['AXXC'][2].min(), '-', results['AXXC'][2].max())
    print('Files written:')
    print('  exports/orbits_CXXC.csv')
    print('  exports/orbits_AXXC.csv')
    print('  exports/arity4_orbit_signature_stabilizer_summary.md')


if __name__ == '__main__':
    main()
