from dataclasses import dataclass
from typing import Tuple

@dataclass
class U:
    r: int
    s: int

@dataclass
class V:
    t: int
    u: int

@dataclass
class W:
    r: int
    u: int

@dataclass
class P:
    left: Tuple[int, int]
    right: Tuple[int, int]

def u_encode(r: int, s: int) -> int:
    return 3 * r + s

def u_decode(u_idx: int) -> U:
    r, s = divmod(u_idx, 3)
    return U(r, s)

def v_encode(t: int, u: int) -> int:
    return 3 * t + u

def v_decode(v_idx: int) -> V:
    t, u = divmod(v_idx, 3)
    return V(t, u)

def w_encode(r: int, u: int) -> int:
    return 3 * r + u

def w_decode(w_idx: int) -> W:
    r, u = divmod(w_idx, 3)
    return W(r, u)

def p_encode(r: int, s: int, t: int, u: int) -> int:
    u_idx = u_encode(r, s)
    v_idx = v_encode(t, u)
    return 9 * u_idx + v_idx

def p_decode(p_idx: int) -> P:
    u_idx, v_idx = divmod(p_idx, 9)
    left = (u_decode(u_idx).r, u_decode(u_idx).s)
    right = (v_decode(v_idx).t, v_decode(v_idx).u)
    return P(left, right)

def build_basis_tables():
    u_table = [u_decode(i) for i in range(9)]
    v_table = [v_decode(i) for i in range(9)]
    w_table = [w_decode(i) for i in range(9)]
    p_table = [p_decode(i) for i in range(81)]
    return u_table, v_table, w_table, p_table

def build_selection_map(p_table):
    selection = []
    for p in p_table:
        r, s = p.left
        t, u = p.right
        if s == t:
            w_idx = w_encode(r, u)
        else:
            w_idx = -1
        selection.append(w_idx)
    return selection

def build_fibers(selection):
    fibers = {i: [] for i in range(9)}
    for p_idx, w_idx in enumerate(selection):
        if w_idx >= 0:
            fibers[w_idx].append(p_idx)
    return fibers

def print_report(u_table, v_table, w_table, p_table, selection, fibers):
    live_count = sum(1 for w in selection if w >= 0)
    dead_count = len(selection) - live_count

    print("=" * 60)
    print("ADE3x3 Step 1: Ambient Bilinear Universe Report")
    print("=" * 60)
    print("\n1. DIMENSIONS")
    print(f"   |U| = {len(u_table)}")
    print(f"   |V| = {len(v_table)}")
    print(f"   |W| = {len(w_table)}")
    print(f"   |P| = {len(p_table)}")

    print("\n2. LIVE/DEAD SPLIT")
    print(f"   live P atoms = {live_count}")
    print(f"   dead P atoms = {dead_count}")

    print("\n3. FIBERS")
    for w_idx in range(9):
        w = w_table[w_idx]
        p_indices = fibers[w_idx]
        p_reprs = [f"P[{p_decode(pi).left[0]},{p_decode(pi).left[1]}|{p_decode(pi).right[0]},{p_decode(pi).right[1]}]" for pi in p_indices]
        print(f"   W[{w.r},{w.u}] <-- {p_reprs}")

    print("\n4. SAMPLE DEAD ATOMS (first 5)")
    dead_examples = []
    for i, w in enumerate(selection):
        if w < 0:
            p = p_table[i]
            dead_examples.append((i, p))
    for p_idx, p in dead_examples[:5]:
        print(f"   P[{p.left[0]},{p.left[1]}|{p.right[0]},{p.right[1]}] --> 0")

    print("\n5. BASIS SAMPLE")
    print(f"   U[1,2] -> idx {u_encode(1,2)}")
    print(f"   V[0,2] -> idx {v_encode(0,2)}")
    print(f"   W[2,1] -> idx {w_encode(2,1)}")
    print(f"   P[1,0|0,2] -> idx {p_encode(1,0,0,2)}")

    print("\n" + "=" * 60)

def main():
    u_table, v_table, w_table, p_table = build_basis_tables()
    selection = build_selection_map(p_table)
    fibers = build_fibers(selection)
    print_report(u_table, v_table, w_table, p_table, selection, fibers)

if __name__ == "__main__":
    main()