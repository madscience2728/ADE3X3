from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parent
CANON = ROOT / 'ADE3x3_CANONICAL_OBJECT.md'
OUT = ROOT / 'ADE3x3_CANONICAL_OBJECT.md'

# ---------- group / naming helpers ----------
S3 = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
ACTIONS = [(ra, sh, cb) for ra in S3 for sh in S3 for cb in S3]


def a_name(a: int) -> str:
    r, s = divmod(a, 3)
    return f"A[{r},{s}]"


def b_name(b: int) -> str:
    t, u = divmod(b, 3)
    return f"B[{t},{u}]"


def c_name(c: int) -> str:
    r, u = divmod(c, 3)
    return f"C[{r},{u}]"


def x_slots(x: int):
    r, s = divmod(x // 9, 3)
    t, u = divmod(x % 9, 3)
    return r, s, t, u


def x_name(x: int) -> str:
    r, s, t, u = x_slots(x)
    return f"X[{r},{s}|{t},{u}]"


def apply_CC(cfg, act):
    c1, c2 = divmod(cfg, 9)
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return 9 * (3 * act[0][r1] + act[2][u1]) + (3 * act[0][r2] + act[2][u2])


def apply_CX(cfg, act):
    c, x = divmod(cfg, 81)
    r, u = divmod(c, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    c_new = 3 * act[0][r] + act[2][u]
    x_new = 9 * (3 * act[0][r2] + act[1][s]) + (3 * act[1][t] + act[2][u2])
    return c_new * 81 + x_new


def apply_XC(cfg, act):
    x, c = divmod(cfg, 9)
    r, s = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    r2, u2 = divmod(c, 3)
    x_new = 9 * (3 * act[0][r] + act[1][s]) + (3 * act[1][t] + act[2][u])
    c_new = 3 * act[0][r2] + act[2][u2]
    return x_new * 9 + c_new


def apply_AX(cfg, act):
    a, x = divmod(cfg, 81)
    r, s = divmod(a, 3)
    r2, s2 = divmod(x // 9, 3); t, u = divmod(x % 9, 3)
    a_new = 3 * act[0][r] + act[1][s]
    x_new = 9 * (3 * act[0][r2] + act[1][s2]) + (3 * act[1][t] + act[2][u])
    return a_new * 81 + x_new


def apply_BX(cfg, act):
    b, x = divmod(cfg, 81)
    t, u = divmod(b, 3)
    r2, s2 = divmod(x // 9, 3); t2, u2 = divmod(x % 9, 3)
    b_new = 3 * act[1][t] + act[2][u]
    x_new = 9 * (3 * act[0][r2] + act[1][s2]) + (3 * act[1][t2] + act[2][u2])
    return b_new * 81 + x_new


def apply_CXC(cfg, act):
    c1 = cfg // (81 * 9)
    rem = cfg % (81 * 9)
    x = rem // 9
    c2 = rem % 9
    r1, u1 = divmod(c1, 3)
    r2, s = divmod(x // 9, 3); t, u2 = divmod(x % 9, 3)
    r3, u3 = divmod(c2, 3)
    c1_new = 3 * act[0][r1] + act[2][u1]
    x_new = 9 * (3 * act[0][r2] + act[1][s]) + (3 * act[1][t] + act[2][u2])
    c2_new = 3 * act[0][r3] + act[2][u3]
    return (c1_new * 81 + x_new) * 9 + c2_new


def apply_XX(cfg, act):
    x1, x2 = divmod(cfg, 81)
    r1, s1 = divmod(x1 // 9, 3); t1, u1 = divmod(x1 % 9, 3)
    r2, s2 = divmod(x2 // 9, 3); t2, u2 = divmod(x2 % 9, 3)
    x1_new = 9 * (3 * act[0][r1] + act[1][s1]) + (3 * act[1][t1] + act[2][u1])
    x2_new = 9 * (3 * act[0][r2] + act[1][s2]) + (3 * act[1][t2] + act[2][u2])
    return x1_new * 81 + x2_new

SCHEMA_META = {
    'CC': (81, lambda cfg: f"CC[{c_name(cfg//9)},{c_name(cfg%9)}]", apply_CC),
    'CX': (729, lambda cfg: f"CX[{c_name(cfg//81)},{x_name(cfg%81)}]", apply_CX),
    'XC': (729, lambda cfg: f"XC[{x_name(cfg//9)},{c_name(cfg%9)}]", apply_XC),
    'AX': (729, lambda cfg: f"AX[{a_name(cfg//81)},{x_name(cfg%81)}]", apply_AX),
    'BX': (729, lambda cfg: f"BX[{b_name(cfg//81)},{x_name(cfg%81)}]", apply_BX),
    'CXC': (6561, lambda cfg: (
        lambda c1, x, c2: f"CXC[{c_name(c1)},{x_name(x)},{c_name(c2)}]"
    )(cfg // (81*9), (cfg % (81*9)) // 9, cfg % 9), apply_CXC),
    'XX': (6561, lambda cfg: f"XX[{x_name(cfg//81)},{x_name(cfg%81)}]", apply_XX),
}


def slots_from_cfg(schema: str, cfg: int):
    if schema == 'CC':
        return (cfg // 9, cfg % 9)
    if schema == 'CX':
        return (cfg // 81, cfg % 81)
    if schema == 'XC':
        return (cfg // 9, cfg % 9)
    if schema == 'AX':
        return (cfg // 81, cfg % 81)
    if schema == 'BX':
        return (cfg // 81, cfg % 81)
    if schema == 'CXC':
        return (cfg // (81*9), (cfg % (81*9)) // 9, cfg % 9)
    if schema == 'XX':
        return (cfg // 81, cfg % 81)
    return ()


def sig_xx(slots):
    x1, x2 = slots
    r1, s1, t1, u1 = x_slots(x1)
    r2, s2, t2, u2 = x_slots(x2)
    live1 = s1 == t1; live2 = s2 == t2
    return (live1, live2, r1 == r2, s1 == s2, t1 == t2, u1 == u2,
            (r1 == r2 and s1 == s2), (t1 == t2 and u1 == u2))


def sig_cx(slots):
    c, x = slots
    cr, cu = divmod(c, 3)
    r, s, t, u = x_slots(x)
    live = s == t
    return (live, live and c == 3*r + u, cr == r, cu == u)


def sig_xc(slots):
    x, c = slots
    r, s, t, u = x_slots(x)
    cr, cu = divmod(c, 3)
    live = s == t
    return (live, live and c == 3*r + u, r == cr, u == cu)


def sig_cc(slots):
    c1, c2 = slots
    r1, u1 = divmod(c1, 3); r2, u2 = divmod(c2, 3)
    return (c1 == c2, r1 == r2, u1 == u2)


def sig_ax(slots):
    a, x = slots
    ar, as_ = divmod(a, 3)
    r, s, t, u = x_slots(x)
    live = s == t
    return (ar == r, as_ == s, live)


def sig_bx(slots):
    b, x = slots
    bt, bu = divmod(b, 3)
    r, s, t, u = x_slots(x)
    live = s == t
    return (bt == t, bu == u, live)


def sig_cxc(slots):
    c1, x, c2 = slots
    r1, u1 = divmod(c1, 3)
    r, s, t, u = x_slots(x)
    r2, u2 = divmod(c2, 3)
    live = s == t
    return (live, c1 == c2, live and c1 == 3*r + u, live and c2 == 3*r + u, (r1, r, r2), (u1, u, u2))

SIG = {'XX': sig_xx, 'CX': sig_cx, 'XC': sig_xc, 'CC': sig_cc, 'AX': sig_ax, 'BX': sig_bx, 'CXC': sig_cxc}
SIG_DESC = {
    'XX': 'Signature fields: (x1_live, x2_live, same_r, same_s, same_t, same_u, same_A_atom, same_B_atom).',
    'CX': 'Signature fields: (x_live, c_is_target_of_x_if_live, c_row_equals_x_row, c_col_equals_x_output_col).',
    'XC': 'Signature fields: (x_live, c_is_target_of_x_if_live, x_row_equals_c_row, x_output_col_equals_c_col).',
    'CC': 'Signature fields: (same_cell, same_row, same_col).',
    'AX': 'Signature fields: (a_row_equals_x_left_row, a_col_equals_x_left_col, x_live).',
    'BX': 'Signature fields: (b_row_equals_x_right_row, b_col_equals_x_right_col, x_live).',
    'CXC': 'Signature fields: (x_live, c1_equals_c2, c1_is_target_if_live, c2_is_target_if_live, row_triple(c1,x,c2), col_triple(c1,x,c2)).',
}


def compute_orbits(schema: str):
    n, name_fn, act_fn = SCHEMA_META[schema]
    reps = {}
    for cfg in range(n):
        m = cfg
        for act in ACTIONS:
            img = act_fn(cfg, act)
            if img < m:
                m = img
        reps[cfg] = m
    groups = defaultdict(list)
    for cfg, rep in reps.items():
        groups[rep].append(cfg)
    rows = []
    for oid, rep in enumerate(sorted(groups)):
        members = groups[rep]
        stab = sum(1 for act in ACTIONS if act_fn(rep, act) == rep)
        sig = SIG[schema](slots_from_cfg(schema, rep))
        rows.append({
            'orbit_id': oid,
            'rep_config_id': rep,
            'rep_readable': name_fn(rep),
            'orbit_size': len(members),
            'stabilizer_size': stab,
            'signature_key': sig,
        })
    return rows


def md_table(headers, rows):
    out = []
    out.append('| ' + ' | '.join(headers) + ' |')
    out.append('|' + '|'.join(['---']*len(headers)) + '|')
    for row in rows:
        out.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(out)


def build_atomic_inverse_tables():
    a_rows = []
    for a in range(9):
        xs = [x_name(9*a + y) for y in range(9)]
        a_rows.append((a, a_name(a), ', '.join(xs)))
    b_rows = []
    # by fixing B=(t,u), vary left A=(r,s)
    for b in range(9):
        t, u = divmod(b, 3)
        xs = []
        for r in range(3):
            for s in range(3):
                xs.append(x_name(9*(3*r + s) + (3*t + u)))
        b_rows.append((b, b_name(b), ', '.join(xs)))
    return a_rows, b_rows


def build_x_live_dead_summary():
    live = 0; dead = 0
    grid = [[0]*3 for _ in range(3)]
    for x in range(81):
        _, s, t, _ = x_slots(x)
        if s == t:
            live += 1
            grid[s][t] += 1
        else:
            dead += 1
    return live, dead, grid


def append_or_replace_section(text: str, header: str, body: str) -> str:
    # simple append; if header already exists, replace from header to next ## or EOF
    marker = f'## {header}'
    if marker not in text:
        if not text.endswith('\n'):
            text += '\n'
        return text + '\n' + body + '\n'
    lines = text.splitlines()
    start = None
    for i,l in enumerate(lines):
        if l.startswith(marker):
            start = i
            break
    end = len(lines)
    for j in range(start+1, len(lines)):
        if lines[j].startswith('## '):
            end = j
            break
    new_lines = lines[:start] + body.splitlines() + lines[end:]
    return '\n'.join(new_lines) + '\n'


def main():
    text = CANON.read_text(encoding='utf-8') if CANON.exists() else ''
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # sections
    a_rows, b_rows = build_atomic_inverse_tables()
    live, dead, grid = build_x_live_dead_summary()

    sec_atomic = []
    sec_atomic += ['## 4A. INLINE LIVE/DEAD AND INVERSE-FIBER TABLES', '', '[GROUND_TRUTH] / [MEASURED_FROM_CODE]', '']
    sec_atomic += ['### X Live/Dead Summary', '', f'- |X_live| = {live}', f'- |X_dead| = {dead}', '', 'Live-count cross-table by (s,t):', '']
    sec_atomic += [md_table(['s\\t','0','1','2'], [[i,*grid[i]] for i in range(3)]), '']
    sec_atomic += ['### A Inverse-Fiber Table', '', md_table(['a_local_id','A atom','X atoms with this a_idx'], a_rows), '']
    sec_atomic += ['### B Inverse-Fiber Table', '', md_table(['b_local_id','B atom','X atoms with this b_idx'], b_rows), '']

    sec_group = []
    sec_group += ['## 5A. GROUP ACTION SPECIFICATION', '', '[GROUND_TRUTH]', '',
                  'The symmetry group is coordinatized by a chosen labeling convention for its three S3 factors:',
                  '- `pi_rA` acts on row indices of A, C, and the left row index of X',
                  '- `pi_shared` acts on the column index of A and both shared middle indices of X and B',
                  '- `pi_cB` acts on column indices of B, C, and the right column index of X',
                  '', 'Compatibility is enforced by using the same `pi_shared` on the A-column / B-row interface.', '',
                  'Generic action on an X atom:', '', '`X[r,s|t,u] -> X[pi_rA(r), pi_shared(s) | pi_shared(t), pi_cB(u)]`', '',
                  'This coordinatization is a naming convention for the three factors, not additional structure beyond the action itself.', '']

    core_schemas = ['CC','CX','XC','AX','BX','XX','CXC']
    orbit_sections = ['## 21. INLINE ORBIT REPRESENTATIVE ROSTERS', '', '[MEASURED_FROM_CODE]', '',
                      'One row per orbit for the core arity-2 and arity-3 schemas.']
    sig_sections = ['## 20. SIGNATURE FORMAT DEFINITIONS', '', '[GROUND_TRUTH] / [MEASURED_FROM_CODE]', '']
    for schema in core_schemas:
        orbits = compute_orbits(schema)
        sig_sections += [f'### {schema} Signature Definition', '', SIG_DESC[schema], '']
        headers = ['orbit_id','rep_config_id','representative','orbit_size','stabilizer_size','signature_key']
        rows = [(r['orbit_id'], r['rep_config_id'], r['rep_readable'], r['orbit_size'], r['stabilizer_size'], repr(r['signature_key'])) for r in orbits]
        orbit_sections += [f'### {schema} Orbit Representatives', '', md_table(headers, rows), '']

    sec_prov = ['## 16A. SUPSERSEDED / REPAIRED STATUS NOTES', '', '[REPAIRED] / [MEASURED_FROM_CODE]', '',
                'The older type-based composition summary `256 / 64 / 36 / 28` is a superseded historical result from an older layer and should not be used as the current canonical composition fact.', '',
                'The earlier BX orbit count `18` is superseded; the corrected current value is `10` after repair of the B-action path.', '']

    # replace agent wording if present
    text = text.replace('## 14. INSTRUCTIONS FOR DOWNSTREAM AGENTS', '## 14. HOW TO READ THIS DOSSIER')
    text = text.replace('### Usage Guidelines', '### Reading Rules')
    text = text.replace('### Basis for Reasoning', '### Scope Reminder')

    text = append_or_replace_section(text, '4A. INLINE LIVE/DEAD AND INVERSE-FIBER TABLES', '\n'.join(sec_atomic))
    text = append_or_replace_section(text, '5A. GROUP ACTION SPECIFICATION', '\n'.join(sec_group))
    text = append_or_replace_section(text, '20. SIGNATURE FORMAT DEFINITIONS', '\n'.join(sig_sections))
    text = append_or_replace_section(text, '21. INLINE ORBIT REPRESENTATIVE ROSTERS', '\n'.join(orbit_sections))
    text = append_or_replace_section(text, '16A. SUPSERSEDED / REPAIRED STATUS NOTES', '\n'.join(sec_prov))

    OUT.write_text(text, encoding='utf-8')
    print(f'Wrote {OUT}')
    print(f'Timestamp: {timestamp}')
    print('Added/updated: inverse-fiber tables, live/dead summary, group action formula, signature definitions, orbit rep rosters')

if __name__ == '__main__':
    main()
