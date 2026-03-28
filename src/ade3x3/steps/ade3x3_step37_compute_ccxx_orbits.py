"""
ade3x3_step37_compute_ccxx_orbits.py

Compute orbits and base signatures for the CCXX schema.

CCXX = C x C x X x X
- Typed arity: 4
- Raw arity: 6  (C, C, A_X1, B_X1, A_X2, B_X2)
- Total configs: 9 x 9 x 81 x 81 = 531,441
"""

from __future__ import annotations

import csv
from collections import defaultdict
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


def _build_actions():
    return [
        (pi_rA, pi_shared, pi_cB)
        for pi_rA in S3
        for pi_shared in S3
        for pi_cB in S3
    ]


ACTIONS = _build_actions()
N_CONFIGS = 9 * 9 * 81 * 81


def decode_CCXX(cfg):
    x2 = cfg % 81
    rem = cfg // 81
    x1 = rem % 81
    rem = rem // 81
    c2 = rem % 9
    c1 = rem // 9

    r1, u1 = divmod(c1, 3)
    r2, u2 = divmod(c2, 3)
    r3, s3 = divmod(x1 // 9, 3)
    t3, u3 = divmod(x1 % 9, 3)
    r4, s4 = divmod(x2 // 9, 3)
    t4, u4 = divmod(x2 % 9, 3)

    return {
        'c1': c1,
        'c2': c2,
        'x1': x1,
        'x2': x2,
        'r1': r1,
        'u1': u1,
        'r2': r2,
        'u2': u2,
        'r3': r3,
        's3': s3,
        't3': t3,
        'u3': u3,
        'r4': r4,
        's4': s4,
        't4': t4,
        'u4': u4,
    }


def get_name_CCXX(cfg):
    d = decode_CCXX(cfg)
    return (
        f"CCXX[C[{d['r1']},{d['u1']}],"
        f"C[{d['r2']},{d['u2']}],"
        f"X[{d['r3']},{d['s3']}|{d['t3']},{d['u3']}],"
        f"X[{d['r4']},{d['s4']}|{d['t4']},{d['u4']}]]"
    )


def act_CCXX(cfg, act):
    d = decode_CCXX(cfg)
    c1n = 3 * act[0][d['r1']] + act[2][d['u1']]
    c2n = 3 * act[0][d['r2']] + act[2][d['u2']]
    x1n = 9 * (3 * act[0][d['r3']] + act[1][d['s3']]) + (
        3 * act[1][d['t3']] + act[2][d['u3']]
    )
    x2n = 9 * (3 * act[0][d['r4']] + act[1][d['s4']]) + (
        3 * act[1][d['t4']] + act[2][d['u4']]
    )
    return ((c1n * 9 + c2n) * 81 + x1n) * 81 + x2n


def _canonical_partition(vals):
    mapping = {}
    next_id = 0
    result = []
    for value in vals:
        if value not in mapping:
            mapping[value] = next_id
            next_id += 1
        result.append(mapping[value])
    return tuple(result)


def sig_CCXX(cfg):
    d = decode_CCXX(cfg)

    x1_live = d['s3'] == d['t3']
    x2_live = d['s4'] == d['t4']
    x1_target = 3 * d['r3'] + d['u3']
    x2_target = 3 * d['r4'] + d['u4']

    row_quad = _canonical_partition((d['r1'], d['r2'], d['r3'], d['r4']))
    col_quad = _canonical_partition((d['u1'], d['u2'], d['u3'], d['u4']))

    return (
        x1_live,
        x2_live,
        d['c1'] == d['c2'],
        x1_live and d['c1'] == x1_target,
        x2_live and d['c1'] == x2_target,
        x1_live and d['c2'] == x1_target,
        x2_live and d['c2'] == x2_target,
        row_quad,
        col_quad,
    )


def canonical_rep_CCXX(cfg):
    d = decode_CCXX(cfg)
    r1 = d['r1']
    u1 = d['u1']
    r2 = d['r2']
    u2 = d['u2']
    r3 = d['r3']
    s3 = d['s3']
    t3 = d['t3']
    u3 = d['u3']
    r4 = d['r4']
    s4 = d['s4']
    t4 = d['t4']
    u4 = d['u4']

    minimum = cfg
    for pi_rA, pi_shared, pi_cB in ACTIONS:
        c1n = 3 * pi_rA[r1] + pi_cB[u1]
        c2n = 3 * pi_rA[r2] + pi_cB[u2]
        x1n = 9 * (3 * pi_rA[r3] + pi_shared[s3]) + (3 * pi_shared[t3] + pi_cB[u3])
        x2n = 9 * (3 * pi_rA[r4] + pi_shared[s4]) + (3 * pi_shared[t4] + pi_cB[u4])
        image = ((c1n * 9 + c2n) * 81 + x1n) * 81 + x2n
        if image < minimum:
            minimum = image
    return minimum


def _compute_chunk(start, stop):
    counts = defaultdict(int)
    for cfg in range(start, stop):
        counts[canonical_rep_CCXX(cfg)] += 1
    return dict(counts)


def _merge_counts(total_counts, chunk_counts):
    for rep, size in chunk_counts.items():
        total_counts[rep] += size


def _iter_chunks(n_configs, chunk_size):
    for start in range(0, n_configs, chunk_size):
        yield start, min(start + chunk_size, n_configs)


def compute_orbits(workers=30, chunk_size=4096):
    print(f"\nComputing orbits for CCXX ({N_CONFIGS:,} configs)...")
    print(f"Using {workers} worker(s) with chunk size {chunk_size:,}\n")

    print("[1/3] Computing canonical representatives...")
    rep_counts = defaultdict(int)

    if workers <= 1:
        processed = 0
        for start, stop in _iter_chunks(N_CONFIGS, chunk_size):
            _merge_counts(rep_counts, _compute_chunk(start, stop))
            processed += stop - start
            pct = 100 * processed / N_CONFIGS
            print(f"  {pct:5.1f}% ({processed:,} / {N_CONFIGS:,})")
    else:
        chunks = list(_iter_chunks(N_CONFIGS, chunk_size))
        processed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_compute_chunk, start, stop): (start, stop)
                for start, stop in chunks
            }
            for future in as_completed(futures):
                start, stop = futures[future]
                _merge_counts(rep_counts, future.result())
                processed += stop - start
                pct = 100 * processed / N_CONFIGS
                print(f"  {pct:5.1f}% ({processed:,} / {N_CONFIGS:,})")

    print("\n[2/3] Grouping orbit representatives...")
    print(f"  Found {len(rep_counts)} orbits")

    print("\n[3/3] Computing signatures and stabilizers...")
    orbit_data = []
    for orbit_id, rep in enumerate(sorted(rep_counts)):
        orbit_size = rep_counts[rep]
        stabilizer_size = sum(1 for act in ACTIONS if act_CCXX(rep, act) == rep)
        signature_key = sig_CCXX(rep)
        orbit_data.append(
            {
                'orbit_id': orbit_id,
                'rep_config_id': rep,
                'rep_readable': get_name_CCXX(rep),
                'signature_key': signature_key,
                'orbit_size': orbit_size,
                'stabilizer_size': stabilizer_size,
            }
        )
        if (orbit_id + 1) % 100 == 0:
            print(f"  {orbit_id + 1} orbits processed...")

    print(f"  {len(orbit_data)} orbits complete\n")

    print("Verifying orbit-stabilizer theorem...")
    for orbit in orbit_data:
        assert orbit['orbit_size'] * orbit['stabilizer_size'] == 216, (
            f"Orbit {orbit['orbit_id']}: "
            f"{orbit['orbit_size']}*{orbit['stabilizer_size']} != 216"
        )
    print("  [OK] All orbits satisfy |orbit| x |stabilizer| = 216\n")

    return orbit_data


def write_csv(orbit_data, path):
    fieldnames = [
        'schema',
        'orbit_id',
        'rep_config_id',
        'rep_readable',
        'signature_key',
        'orbit_size',
        'stabilizer_size',
    ]

    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in orbit_data:
            writer.writerow(
                {
                    'schema': 'CCXX',
                    'orbit_id': row['orbit_id'],
                    'rep_config_id': row['rep_config_id'],
                    'rep_readable': row['rep_readable'],
                    'signature_key': repr(row['signature_key']),
                    'orbit_size': row['orbit_size'],
                    'stabilizer_size': row['stabilizer_size'],
                }
            )


def write_summary_md(orbit_data, path):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_orbits = len(orbit_data)
    sig_groups = defaultdict(int)
    for orbit in orbit_data:
        sig_groups[repr(orbit['signature_key'])] += 1
    n_sigs = len(sig_groups)
    collision_groups = sum(1 for size in sig_groups.values() if size > 1)
    collisions = sum(size for size in sig_groups.values() if size > 1)
    total_configs = sum(orbit['orbit_size'] for orbit in orbit_data)

    lines = [
        '# CCXX Orbit Computation Summary',
        '',
        f'**Generated:** {ts}',
        '**Schema:** CCXX (C x C x X x X)',
        '**Typed Arity:** 4',
        '**Raw Arity:** 6',
        '',
        '## Results',
        '',
        f'- Total configurations: {total_configs:,}',
        f'- Total orbits: {n_orbits}',
        f'- Distinct signatures: {n_sigs}',
        f'- Collision groups: {collision_groups}',
        f'- Collision count: {collisions}',
        f"- Orbit-complete: {'yes' if n_sigs == n_orbits else 'no'}",
        '',
        '## Verification',
        '',
        '- All orbits satisfy |orbit| x |stabilizer| = 216 ✓',
        f'- Sum of orbit sizes = {total_configs:,} ✓',
        '',
        '## Signature Format',
        '',
        'Signature tuple fields:',
        '```',
        '(x1_live, x2_live, c1_equals_c2,',
        ' c1_is_target_of_x1_if_live, c1_is_target_of_x2_if_live,',
        ' c2_is_target_of_x1_if_live, c2_is_target_of_x2_if_live,',
        ' row_quad, col_quad)',
        '```',
        '',
        'Where:',
        '- x1_live, x2_live: Boolean liveness of each X atom',
        '- c1_equals_c2: Whether the two C atoms are equal',
        '- Target flags: Whether a live X atom targets one of the C atoms',
        '- row_quad, col_quad: Canonical equality partitions of rows and columns',
        '',
        '## Exported Files',
        '',
        '- signatures_CCXX.csv: Complete orbit roster with signatures',
        '- ccxx_computation_summary.md: This file',
        '',
    ]

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))


def main():
    import argparse
    import multiprocessing
    import time

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Compute CCXX orbits and signatures.')
    parser.add_argument('--workers', type=int, default=30, help='Process workers for canonical-rep computation.')
    parser.add_argument('--chunk-size', type=int, default=4096, help='Configurations per submitted chunk.')
    args = parser.parse_args()

    print('=' * 70)
    print('CCXX Orbit Computation')
    print('=' * 70)
    print(f'Group size: {len(ACTIONS)}')

    start = time.time()
    orbit_data = compute_orbits(workers=args.workers, chunk_size=args.chunk_size)
    elapsed = time.time() - start

    exports = Path('outputs/exports')
    exports.mkdir(parents=True, exist_ok=True)

    csv_path = exports / 'signatures_CCXX.csv'
    write_csv(orbit_data, csv_path)
    print(f'Written: {csv_path}')

    md_path = exports / 'ccxx_computation_summary.md'
    write_summary_md(orbit_data, md_path)
    print(f'Written: {md_path}')

    n_orbits = len(orbit_data)
    n_sigs = len({repr(orbit['signature_key']) for orbit in orbit_data})

    print('\n' + '=' * 70)
    print('COMPUTATION COMPLETE')
    print('=' * 70)
    print(f'Orbits: {n_orbits}')
    print(f'Distinct signatures: {n_sigs}')
    print(f'Time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)')
    print('=' * 70)


if __name__ == '__main__':
    main()