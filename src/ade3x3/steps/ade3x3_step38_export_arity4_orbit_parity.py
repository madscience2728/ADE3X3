"""
ade3x3_step38_export_arity4_orbit_parity.py

Recompute arity-4 orbit and stabilizer data for the first two raw-arity-6
schemas and compare those orbit counts against the currently exported
arity-4 signature layers.

Schemas:
- CXXC = C x X x X x C  -> raw slots (C, A_X1, B_X1, A_X2, B_X2, C)
- AXXC = A x X x X x C  -> raw slots (A, A_X1, B_X1, A_X2, B_X2, C)
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


S3 = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]

POW9 = (9**5, 9**4, 9**3, 9**2, 9, 1)
N_CONFIGS = 9**6
EXPORTS = Path('outputs/exports')


def _build_actions():
    actions = []
    for pi_rA in S3:
        for pi_shared in S3:
            for pi_cB in S3:
                a_map = []
                b_map = []
                c_map = []
                for idx in range(9):
                    r, s = divmod(idx, 3)
                    a_map.append(3 * pi_rA[r] + pi_shared[s])
                for idx in range(9):
                    t, u = divmod(idx, 3)
                    b_map.append(3 * pi_shared[t] + pi_cB[u])
                for idx in range(9):
                    r, u = divmod(idx, 3)
                    c_map.append(3 * pi_rA[r] + pi_cB[u])
                actions.append((tuple(a_map), tuple(b_map), tuple(c_map)))
    return actions


ACTIONS = _build_actions()
SCHEMAS = {
    'CXXC': ('C', 'A', 'B', 'A', 'B', 'C'),
    'AXXC': ('A', 'A', 'B', 'A', 'B', 'C'),
}
AXXC_FACE_FIELDS = (
    'left_ax_orbit_id',
    'middle_xx_orbit_id',
    'right_xc_orbit_id',
    'outer_ac_orbit_id',
    'left_axc_orbit_id',
    'right_axc_orbit_id',
)


def raw6_id(s0, s1, s2, s3, s4, s5):
    return ((((s0 * 9 + s1) * 9 + s2) * 9 + s3) * 9 + s4) * 9 + s5


def decode_raw6(raw_id):
    vals = []
    rem = raw_id
    for power in POW9:
        digit = rem // power
        vals.append(digit)
        rem %= power
    return tuple(vals)


def c_name(idx):
    return f"C[{idx // 3},{idx % 3}]"


def a_name(idx):
    return f"A[{idx // 3},{idx % 3}]"


def x_name(a_idx, b_idx):
    r, s = divmod(a_idx, 3)
    t, u = divmod(b_idx, 3)
    return f"X[{r},{s}|{t},{u}]"


def readable(schema_name, raw_id):
    v0, v1, v2, v3, v4, v5 = decode_raw6(raw_id)
    if schema_name == 'CXXC':
        return f"CXXC[{c_name(v0)},{x_name(v1, v2)},{x_name(v3, v4)},{c_name(v5)}]"
    return f"AXXC[{a_name(v0)},{x_name(v1, v2)},{x_name(v3, v4)},{c_name(v5)}]"


def act_raw6(raw_id, action, slot_types):
    vals = decode_raw6(raw_id)
    out = []
    a_map, b_map, c_map = action
    for value, slot_type in zip(vals, slot_types):
        if slot_type == 'A':
            out.append(a_map[value])
        elif slot_type == 'B':
            out.append(b_map[value])
        else:
            out.append(c_map[value])
    return raw6_id(*out)


def canonical_rep_raw6(raw_id, slot_types):
    vals = decode_raw6(raw_id)
    minimum = raw_id
    for a_map, b_map, c_map in ACTIONS:
        out = []
        for value, slot_type in zip(vals, slot_types):
            if slot_type == 'A':
                out.append(a_map[value])
            elif slot_type == 'B':
                out.append(b_map[value])
            else:
                out.append(c_map[value])
        image = raw6_id(*out)
        if image < minimum:
            minimum = image
    return minimum


def _iter_chunks(n_configs, chunk_size):
    for start in range(0, n_configs, chunk_size):
        yield start, min(start + chunk_size, n_configs)


def _compute_chunk(schema_name, start, stop):
    slot_types = SCHEMAS[schema_name]
    counts = defaultdict(int)
    for raw_id in range(start, stop):
        counts[canonical_rep_raw6(raw_id, slot_types)] += 1
    return dict(counts)


def _merge_counts(total_counts, chunk_counts):
    for rep, size in chunk_counts.items():
        total_counts[rep] += size


def compute_orbits(schema_name, workers=30, chunk_size=4096):
    print(f"\nComputing {schema_name} raw-arity-6 orbits ({N_CONFIGS:,} configs)...")
    print(f"Using {workers} worker(s) with chunk size {chunk_size:,}")

    rep_counts = defaultdict(int)
    chunks = list(_iter_chunks(N_CONFIGS, chunk_size))
    processed = 0

    if workers <= 1:
        for start, stop in chunks:
            _merge_counts(rep_counts, _compute_chunk(schema_name, start, stop))
            processed += stop - start
            pct = 100 * processed / N_CONFIGS
            print(f"  {pct:5.1f}% ({processed:,} / {N_CONFIGS:,})")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_compute_chunk, schema_name, start, stop): (start, stop)
                for start, stop in chunks
            }
            for future in as_completed(futures):
                start, stop = futures[future]
                _merge_counts(rep_counts, future.result())
                processed += stop - start
                pct = 100 * processed / N_CONFIGS
                print(f"  {pct:5.1f}% ({processed:,} / {N_CONFIGS:,})")

    orbit_data = []
    slot_types = SCHEMAS[schema_name]
    for orbit_id, rep in enumerate(sorted(rep_counts)):
        orbit_size = rep_counts[rep]
        stabilizer_size = sum(1 for action in ACTIONS if act_raw6(rep, action, slot_types) == rep)
        orbit_data.append(
            {
                'schema': schema_name,
                'orbit_id': orbit_id,
                'rep_config_id': rep,
                'rep_readable': readable(schema_name, rep),
                'orbit_size': orbit_size,
                'stabilizer_size': stabilizer_size,
            }
        )
        if (orbit_id + 1) % 250 == 0:
            print(f"  {schema_name}: {orbit_id + 1} orbits processed...")

    for orbit in orbit_data:
        assert orbit['orbit_size'] * orbit['stabilizer_size'] == 216, (
            f"{schema_name} orbit {orbit['orbit_id']}: "
            f"{orbit['orbit_size']}*{orbit['stabilizer_size']} != 216"
        )

    print(f"  Found {len(orbit_data)} {schema_name} orbits")
    print(f"  [OK] All {schema_name} orbits satisfy |orbit| x |stabilizer| = 216")
    return orbit_data


def write_orbits_csv(schema_name, orbit_data, path):
    fieldnames = [
        'schema',
        'orbit_id',
        'rep_config_id',
        'rep_readable',
        'orbit_size',
        'stabilizer_size',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(orbit_data)


def count_cxxc_signature_patterns(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return sum(1 for _ in csv.DictReader(handle))


def count_axxc_signature_patterns(path):
    keys = set()
    with open(path, 'r', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            keys.add(tuple(int(row[field]) for field in AXXC_FACE_FIELDS))
    return len(keys)


def summarize_schema(schema_name, orbit_data, signature_count):
    orbit_sizes = [row['orbit_size'] for row in orbit_data]
    stab_sizes = [row['stabilizer_size'] for row in orbit_data]
    return {
        'schema': schema_name,
        'orbit_count': len(orbit_data),
        'signature_count': signature_count,
        'orbit_complete': len(orbit_data) == signature_count,
        'min_orbit_size': min(orbit_sizes),
        'max_orbit_size': max(orbit_sizes),
        'min_stabilizer_size': min(stab_sizes),
        'max_stabilizer_size': max(stab_sizes),
    }


def write_summary(results, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        '# Arity-4 Orbit / Signature / Stabilizer Parity',
        '',
        f'Generated: {ts}',
        '',
        '| Schema | Orbit Count | Signature Count | Orbit Complete | Min Orbit | Max Orbit | Min Stab | Max Stab |',
        '|--------|-------------|-----------------|----------------|-----------|-----------|----------|----------|',
    ]
    for result in results:
        lines.append(
            f"| {result['schema']} | {result['orbit_count']} | {result['signature_count']} | "
            f"{'yes' if result['orbit_complete'] else 'no'} | {result['min_orbit_size']} | "
            f"{result['max_orbit_size']} | {result['min_stabilizer_size']} | {result['max_stabilizer_size']} |"
        )
    lines += [
        '',
        'Notes:',
        '- Orbit counts and stabilizer ranges were recomputed directly from the 216-element action on raw arity-6 tuples.',
        '- CXXC signature count is the row count of CXXC_face_patterns.csv.',
        '- AXXC signature count is the number of distinct face-pattern tuples induced by AXXC_face_inventory.csv.',
        '- Equality of orbit count and signature count is the current arity-4 parity fact.',
    ]
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Export arity-4 orbit/signature/stabilizer parity for CXXC and AXXC.')
    parser.add_argument('--workers', type=int, default=30, help='Process workers for canonical-rep computation.')
    parser.add_argument('--chunk-size', type=int, default=4096, help='Configurations per submitted chunk.')
    args = parser.parse_args()

    EXPORTS.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('Arity-4 Orbit / Signature / Stabilizer Parity')
    print('=' * 70)
    print(f'Group size: {len(ACTIONS)}')

    cxxc_orbits = compute_orbits('CXXC', workers=args.workers, chunk_size=args.chunk_size)
    axxc_orbits = compute_orbits('AXXC', workers=args.workers, chunk_size=args.chunk_size)

    cxxc_csv = EXPORTS / 'orbits_CXXC.csv'
    axxc_csv = EXPORTS / 'orbits_AXXC.csv'
    write_orbits_csv('CXXC', cxxc_orbits, cxxc_csv)
    write_orbits_csv('AXXC', axxc_orbits, axxc_csv)

    cxxc_signature_count = count_cxxc_signature_patterns(EXPORTS / 'CXXC_face_patterns.csv')
    axxc_signature_count = count_axxc_signature_patterns(EXPORTS / 'AXXC_face_inventory.csv')

    results = [
        summarize_schema('CXXC', cxxc_orbits, cxxc_signature_count),
        summarize_schema('AXXC', axxc_orbits, axxc_signature_count),
    ]

    summary_path = EXPORTS / 'arity4_orbit_signature_stabilizer_summary.md'
    write_summary(results, summary_path)

    print('\nResults:')
    for result in results:
        print(
            f"  {result['schema']}: {result['orbit_count']} orbits; "
            f"signatures = {result['signature_count']}; "
            f"orbit-complete = {'yes' if result['orbit_complete'] else 'no'}; "
            f"stabilizer range {result['min_stabilizer_size']}-{result['max_stabilizer_size']}"
        )

    print('\nFiles written:')
    print(f'  {cxxc_csv}')
    print(f'  {axxc_csv}')
    print(f'  {summary_path}')


if __name__ == '__main__':
    main()