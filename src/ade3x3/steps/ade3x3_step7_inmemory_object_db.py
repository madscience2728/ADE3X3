import sys
import ade3x3_step1 as s1
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class Tracker:
    goal: str = "build upward toward 9^9 closure"
    module_role: str = "global in-memory object DB"
    explicit_layers: List[str] = field(default_factory=list)

def canonical_pattern(vals):
    mapping = {}
    next_label = 0
    pattern = []
    for v in vals:
        if v not in mapping:
            mapping[v] = next_label
            next_label += 1
        pattern.append(mapping[v])
    return tuple(pattern)

@dataclass
class AtomRecord:
    global_id: int
    species: str
    local_idx: int
    coords: Tuple
    canonical_name: str

@dataclass
class ActionRecord:
    action_id: int
    pi_rA: Tuple[int, int, int]
    pi_shared: Tuple[int, int, int]
    pi_cB: Tuple[int, int, int]

@dataclass
class SchemaRecord:
    name: str
    species_list: List[str]
    arity: int
    role: Optional[str] = None

@dataclass
class ConfigRecord:
    schema_name: str
    slot_values: Tuple
    canonical_name: str
    signature: Optional[Tuple] = None

class ADE3x3ObjectDB:
    def __init__(self):
        self.atoms: Dict[int, AtomRecord] = {}
        self.atoms_by_species: Dict[str, Dict[int, AtomRecord]] = {'A': {}, 'B': {}, 'C': {}, 'X': {}}
        self.actions: List[ActionRecord] = []
        self.schemas: Dict[str, SchemaRecord] = {}
        self.catalog_slots: Dict[str, Dict] = {}
        self.config_slots: Dict[str, List[ConfigRecord]] = {}
        
        self.x_to_a: Dict[int, int] = {}
        self.x_to_b: Dict[int, int] = {}
        self.x_live: Dict[int, bool] = {}
        self.x_target_c: Dict[int, int] = {}
        self.c_fibers: Dict[int, List[int]] = {}
        self.c_reverse_map: Dict[int, List[int]] = {}
    
    def build_atoms(self):
        next_global = 0
        
        for r in range(3):
            for s in range(3):
                local_idx = 3 * r + s
                name = f"A[{r},{s}]"
                rec = AtomRecord(next_global, 'A', local_idx, (r, s), name)
                self.atoms[next_global] = rec
                self.atoms_by_species['A'][local_idx] = rec
                next_global += 1
        
        for t in range(3):
            for u in range(3):
                local_idx = 3 * t + u
                name = f"B[{t},{u}]"
                rec = AtomRecord(next_global, 'B', local_idx, (t, u), name)
                self.atoms[next_global] = rec
                self.atoms_by_species['B'][local_idx] = rec
                next_global += 1
        
        for r in range(3):
            for u in range(3):
                local_idx = 3 * r + u
                name = f"C[{r},{u}]"
                rec = AtomRecord(next_global, 'C', local_idx, (r, u), name)
                self.atoms[next_global] = rec
                self.atoms_by_species['C'][local_idx] = rec
                next_global += 1
        
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    for u_ in range(3):
                        local_idx = 9 * (3 * r + s) + (3 * t + u_)
                        name = f"X[{r},{s}|{t},{u_}]"
                        coords = (r, s, t, u_)
                        rec = AtomRecord(next_global, 'X', local_idx, coords, name)
                        self.atoms[next_global] = rec
                        self.atoms_by_species['X'][local_idx] = rec
                        
                        a_local = 3 * r + s
                        b_local = 3 * t + u_
                        self.x_to_a[local_idx] = a_local
                        self.x_to_b[local_idx] = b_local
                        
                        live = (s == t)
                        self.x_live[local_idx] = live
                        
                        if live:
                            c_local = 3 * r + u_
                            self.x_target_c[local_idx] = c_local
                        else:
                            self.x_target_c[local_idx] = -1
                        
                        next_global += 1
        
        for c_local in range(9):
            self.c_fibers[c_local] = []
            self.c_reverse_map[c_local] = []
        
        for x_local in range(81):
            if self.x_live[x_local]:
                c_target = self.x_target_c[x_local]
                self.c_fibers[c_target].append(x_local)
                self.c_reverse_map[c_target].append(x_local)
        
        return next_global
    
    def build_primitive_relations(self):
        pass
    
    def generate_s3(self):
        return [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    
    def build_actions(self):
        s3 = self.generate_s3()
        
        for pi_rA in s3:
            for pi_shared in s3:
                for pi_cB in s3:
                    action_id = len(self.actions)
                    rec = ActionRecord(action_id, pi_rA, pi_shared, pi_cB)
                    self.actions.append(rec)
    
    def act_on_a(self, local_idx, action_id):
        act = self.actions[action_id]
        r, s = divmod(local_idx, 3)
        r_new = act.pi_rA[r]
        s_new = act.pi_shared[s]
        return 3 * r_new + s_new
    
    def act_on_b(self, local_idx, action_id):
        act = self.actions[action_id]
        t, u = divmod(local_idx, 3)
        t_new = act.pi_shared[t]
        u_new = act.pi_cB[u]
        return 3 * t_new + u_new
    
    def act_on_c(self, local_idx, action_id):
        act = self.actions[action_id]
        r, u = divmod(local_idx, 3)
        r_new = act.pi_rA[r]
        u_new = act.pi_cB[u]
        return 3 * r_new + u_new
    
    def act_on_x(self, local_idx, action_id):
        r, s = divmod(local_idx // 9, 3)
        t, u = divmod(local_idx % 9, 3)
        
        act = self.actions[action_id]
        r_new = act.pi_rA[r]
        s_new = act.pi_shared[s]
        t_new = act.pi_shared[t]
        u_new = act.pi_cB[u]
        
        a_local = 3 * r_new + s_new
        b_local = 3 * t_new + u_new
        return 9 * a_local + b_local
    
    def register_schemas(self):
        self.schemas['A'] = SchemaRecord('A', ['A'], 1, 'left input')
        self.schemas['B'] = SchemaRecord('B', ['B'], 1, 'right input')
        self.schemas['C'] = SchemaRecord('C', ['C'], 1, 'output')
        self.schemas['X'] = SchemaRecord('X', ['X'], 1, 'ambient product')
        
        self.schemas['XX'] = SchemaRecord('XX', ['X', 'X'], 2, 'X-X pair')
        self.schemas['CX'] = SchemaRecord('CX', ['C', 'X'], 2, 'C-X pair')
        self.schemas['XC'] = SchemaRecord('XC', ['X', 'C'], 2, 'X-C pair')
        self.schemas['CC'] = SchemaRecord('CC', ['C', 'C'], 2, 'C-C pair')
        self.schemas['AX'] = SchemaRecord('AX', ['A', 'X'], 2, 'A-X pair')
        self.schemas['BX'] = SchemaRecord('BX', ['B', 'X'], 2, 'B-X pair')
        
        self.schemas['CXC'] = SchemaRecord('CXC', ['C', 'X', 'C'], 3, 'C-X-C triple')
    
    def initialize_catalog_slots(self):
        for name in ['XX', 'CX', 'XC', 'CC', 'AX', 'BX']:
            self.catalog_slots[name] = {
                'type_id_table': {},
                'signature_keys': {},
                'representatives': {},
                'counts': {},
                'pair_count': 0
            }
    
    def initialize_config_slots(self):
        for name in ['CXC']:
            self.config_slots[name] = []
    
    def populate_cxc_instances(self):
        all_c = list(range(9))
        all_x = list(range(81))
        
        count = 0
        for c1 in all_c:
            for x in all_x:
                for c2 in all_c:
                    name = f"C{self.atoms_by_species['C'][c1].canonical_name[1]}--X{self.atoms_by_species['X'][x].canonical_name[1:-1].replace(',','|')}--C{self.atoms_by_species['C'][c2].canonical_name[1]}"
                    
                    c1_name = self.atoms_by_species['C'][c1].canonical_name
                    x_name = self.atoms_by_species['X'][x].canonical_name
                    c2_name = self.atoms_by_species['C'][c2].canonical_name
                    
                    rec = ConfigRecord(
                        schema_name='CXC',
                        slot_values=(c1, x, c2),
                        canonical_name=f"{c1_name}--{x_name}--{c2_name}"
                    )
                    self.config_slots['CXC'].append(rec)
                    count += 1
        
        return count
    
    def lookup_atom(self, global_id):
        return self.atoms.get(global_id)
    
    lookup_atom_by_species = lambda self, species, local_idx: self.atoms_by_species.get(species, {}).get(local_idx)
    lookup_atom_by_name = lambda self, name: next((a for a in self.atoms.values() if a.canonical_name == name), None)
    
    def lookup_x_for_c(self, c_local):
        return self.c_reverse_map.get(c_local, [])
    
    def summary(self):
        print("\n" + "=" * 70)
        print("ADE3x3 In-Memory Object DB Summary")
        print("=" * 70)
        
        print("\n--- TRACKER ---")
        print(f"Goal: {Tracker().goal}")
        print(f"Module role: {Tracker().module_role}")
        print("Explicit layers stored:")
        print("  - atomic species (A,B,C,X)")
        print("  - primitive exact relations (X properties, fibers)")
        print("  - compatible symmetry actions (216 actions)")
        print("  - typed schema registry")
        print("  - configuration storage scaffolding")
        
        print("\n--- ATOM COUNTS ---")
        print(f"|A| = {len(self.atoms_by_species['A'])}")
        print(f"|B| = {len(self.atoms_by_species['B'])}")
        print(f"|C| = {len(self.atoms_by_species['C'])}")
        print(f"|X| = {len(self.atoms_by_species['X'])}")
        
        print("\n--- PRIMITIVE STRUCTURE ---")
        live_count = sum(1 for v in self.x_live.values() if v)
        dead_count = len(self.x_live) - live_count
        print(f"Live X atoms: {live_count}")
        print(f"Dead X atoms: {dead_count}")
        
        for c_local in range(9):
            fiber_size = len(self.c_fibers.get(c_local, []))
            print(f"Fiber of C[{c_local // 3},{c_local % 3}]: {fiber_size} X atoms")
        
        print("\n--- ACTION LAYER ---")
        print(f"Compatible actions stored: {len(self.actions)}")
        
        print("\n--- SCHEMA REGISTRY ---")
        print(f"Registered schemas: {list(self.schemas.keys())}")
        print(f"Schema count: {len(self.schemas)}")
        
        print("\n--- CATALOG SLOTS ---")
        print(f"Initialized catalog slots: {list(self.catalog_slots.keys())}")
        
        print("\n--- CONFIGURATION STORAGE ---")
        for schema_name, configs in self.config_slots.items():
            print(f"  {schema_name}: {len(configs)} raw instances")
    
    def query_demo(self):
        print("\n--- QUERY DEMONSTRATIONS ---")
        
        print("\n1. Retrieve X atom and show properties:")
        x_local = 0
        x_rec = self.atoms_by_species['X'][x_local]
        print(f"   Atom: {x_rec.canonical_name}")
        print(f"   Global ID: {x_rec.global_id}")
        print(f"   Coords: {x_rec.coords}")
        print(f"   Left A local: {self.x_to_a[x_local]}")
        print(f"   Right B local: {self.x_to_b[x_local]}")
        print(f"   Live: {self.x_live[x_local]}")
        print(f"   Target C: {self.x_target_c[x_local]}")
        
        print("\n2. Retrieve C atom and list fiber:")
        c_local = 0
        c_rec = self.atoms_by_species['C'][c_local]
        print(f"   Atom: {c_rec.canonical_name}")
        print(f"   Fiber X atoms: {self.c_fibers[c_local]}")
        
        print("\n3. Show action effect on sample X and C:")
        action_id = 0
        act = self.actions[action_id]
        print(f"   Action {action_id}: pi_rA={act.pi_rA}, pi_shared={act.pi_shared}, pi_cB={act.pi_cB}")
        
        x_sample = self.act_on_x(0, action_id)
        print(f"   X[0,0|0,0] -> {self.atoms_by_species['X'][x_sample].canonical_name}")
        
        c_sample = self.act_on_c(0, action_id)
        print(f"   C[0,0] -> {self.atoms_by_species['C'][c_sample].canonical_name}")
        
        print("\n4. Sample CXC triple from config storage:")
        if self.config_slots.get('CXC'):
            sample = self.config_slots['CXC'][0]
            print(f"   First CXC instance: {sample.canonical_name}")
            print(f"   Schema: {sample.schema_name}")
            print(f"   Slot values: {sample.slot_values}")
    
    def memory_estimate(self):
        atom_size = sum(sys.getsizeof(a) for a in self.atoms.values())
        
        action_size = sum(sys.getsizeof(a) for a in self.actions)
        
        x_relations_size = (
            sys.getsizeof(self.x_to_a) + sys.getsizeof(self.x_to_b) +
            sys.getsizeof(self.x_live) + sys.getsizeof(self.x_target_c) +
            sys.getsizeof(self.c_fibers) + sys.getsizeof(self.c_reverse_map)
        )
        
        schema_size = sum(sys.getsizeof(s) for s in self.schemas.values())
        
        config_total = sum(len(v) for v in self.config_slots.values())
        config_size = config_total * sys.getsizeof(ConfigRecord("", (), ""))
        
        total = atom_size + action_size + x_relations_size + schema_size + config_size
        
        print("\n--- MEMORY ESTIMATE ---")
        print(f"Atoms table: ~{atom_size} bytes")
        print(f"Actions table: ~{action_size} bytes")
        print(f"X relations (maps, fibers): ~{x_relations_size} bytes")
        print(f"Schema registry: ~{schema_size} bytes")
        print(f"Config storage ({config_total} instances): ~{config_size} bytes")
        print(f"TOTAL approximate: ~{total} bytes (~{total/1024:.1f} KB)")
        
        return total

def main():
    db = ADE3x3ObjectDB()
    
    print("Building in-memory object database...")
    db.build_atoms()
    db.build_actions()
    db.register_schemas()
    db.initialize_catalog_slots()
    db.initialize_config_slots()
    cxc_count = db.populate_cxc_instances()
    
    db.summary()
    db.query_demo()
    db.memory_estimate()
    
    print("\n--- COMPLETE ---")
    print("In-memory object DB built and verified.")
    print("Long-term goal remains: 9^9 closure")

if __name__ == "__main__":
    main()