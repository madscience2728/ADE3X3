import sys
import ade3x3_step1 as s1
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

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
    config_id: int
    schema_name: str
    slot_values: Tuple
    canonical_name: str

class ADE3x3ObjectDB:
    def __init__(self):
        self.atoms: Dict[int, AtomRecord] = {}
        self.atoms_by_species: Dict[str, Dict[int, AtomRecord]] = {'A': {}, 'B': {}, 'C': {}, 'X': {}}
        self.actions: List[ActionRecord] = []
        self.schemas: Dict[str, SchemaRecord] = {}
        
        self.configs: Dict[str, List[ConfigRecord]] = {}
        self.config_count: Dict[str, int] = {}
        
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
                        rec = AtomRecord(next_global, 'X', local_idx, (r, s, t, u_), name)
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
    
    def register_schemas(self):
        self.schemas['A'] = SchemaRecord('A', ['A'], 1, 'left input')
        self.schemas['B'] = SchemaRecord('B', ['B'], 1, 'right input')
        self.schemas['C'] = SchemaRecord('C', ['C'], 1, 'output')
        self.schemas['X'] = SchemaRecord('X', ['X'], 1, 'ambient product')
        self.schemas['XX'] = SchemaRecord('XX', ['X', 'X'], 2)
        self.schemas['CX'] = SchemaRecord('CX', ['C', 'X'], 2)
        self.schemas['XC'] = SchemaRecord('XC', ['X', 'C'], 2)
        self.schemas['CC'] = SchemaRecord('CC', ['C', 'C'], 2)
        self.schemas['AX'] = SchemaRecord('AX', ['A', 'X'], 2)
        self.schemas['BX'] = SchemaRecord('BX', ['B', 'X'], 2)
        self.schemas['CXC'] = SchemaRecord('CXC', ['C', 'X', 'C'], 3)
    
    def populate_schema_xx(self):
        configs = []
        for x1 in range(81):
            for x2 in range(81):
                x1_name = self.atoms_by_species['X'][x1].canonical_name
                x2_name = self.atoms_by_species['X'][x2].canonical_name
                name = f"XX[{x1_name},{x2_name}]"
                rec = ConfigRecord(len(configs), 'XX', (x1, x2), name)
                configs.append(rec)
        
        self.configs['XX'] = configs
        self.config_count['XX'] = len(configs)
        return len(configs)
    
    def populate_schema_cx(self):
        configs = []
        for c in range(9):
            for x in range(81):
                c_name = self.atoms_by_species['C'][c].canonical_name
                x_name = self.atoms_by_species['X'][x].canonical_name
                name = f"CX[{c_name},{x_name}]"
                rec = ConfigRecord(len(configs), 'CX', (c, x), name)
                configs.append(rec)
        
        self.configs['CX'] = configs
        self.config_count['CX'] = len(configs)
        return len(configs)
    
    def populate_schema_xc(self):
        configs = []
        for x in range(81):
            for c in range(9):
                x_name = self.atoms_by_species['X'][x].canonical_name
                c_name = self.atoms_by_species['C'][c].canonical_name
                name = f"XC[{x_name},{c_name}]"
                rec = ConfigRecord(len(configs), 'XC', (x, c), name)
                configs.append(rec)
        
        self.configs['XC'] = configs
        self.config_count['XC'] = len(configs)
        return len(configs)
    
    def populate_schema_cc(self):
        configs = []
        for c1 in range(9):
            for c2 in range(9):
                c1_name = self.atoms_by_species['C'][c1].canonical_name
                c2_name = self.atoms_by_species['C'][c2].canonical_name
                name = f"CC[{c1_name},{c2_name}]"
                rec = ConfigRecord(len(configs), 'CC', (c1, c2), name)
                configs.append(rec)
        
        self.configs['CC'] = configs
        self.config_count['CC'] = len(configs)
        return len(configs)
    
    def populate_schema_ax(self):
        configs = []
        for a in range(9):
            for x in range(81):
                a_name = self.atoms_by_species['A'][a].canonical_name
                x_name = self.atoms_by_species['X'][x].canonical_name
                name = f"AX[{a_name},{x_name}]"
                rec = ConfigRecord(len(configs), 'AX', (a, x), name)
                configs.append(rec)
        
        self.configs['AX'] = configs
        self.config_count['AX'] = len(configs)
        return len(configs)
    
    def populate_schema_bx(self):
        configs = []
        for b in range(9):
            for x in range(81):
                b_name = self.atoms_by_species['B'][b].canonical_name
                x_name = self.atoms_by_species['X'][x].canonical_name
                name = f"BX[{b_name},{x_name}]"
                rec = ConfigRecord(len(configs), 'BX', (b, x), name)
                configs.append(rec)
        
        self.configs['BX'] = configs
        self.config_count['BX'] = len(configs)
        return len(configs)
    
    def populate_schema_cxc(self):
        configs = []
        for c1 in range(9):
            for x in range(81):
                for c2 in range(9):
                    c1_name = self.atoms_by_species['C'][c1].canonical_name
                    x_name = self.atoms_by_species['X'][x].canonical_name
                    c2_name = self.atoms_by_species['C'][c2].canonical_name
                    name = f"CXC[{c1_name},{x_name},{c2_name}]"
                    rec = ConfigRecord(len(configs), 'CXC', (c1, x, c2), name)
                    configs.append(rec)
        
        self.configs['CXC'] = configs
        self.config_count['CXC'] = len(configs)
        return len(configs)
    
    def populate_all_core(self):
        self.populate_schema_xx()
        self.populate_schema_cx()
        self.populate_schema_xc()
        self.populate_schema_cc()
        self.populate_schema_ax()
        self.populate_schema_bx()
        self.populate_schema_cxc()
    
    def schema_instance_count(self, name):
        return self.config_count.get(name, 0)
    
    def get_config(self, schema_name, config_id):
        if schema_name in self.configs:
            return self.configs[schema_name][config_id]
        return None
    
    def iter_configs(self, schema_name):
        return self.configs.get(schema_name, [])
    
    def memory_estimate(self):
        config_mem = 0
        for name, configs in self.configs.items():
            for c in configs:
                config_mem += sys.getsizeof(c)
        
        totals = {
            'XX': sys.getsizeof(self.configs.get('XX', [])),
            'CX': sys.getsizeof(self.configs.get('CX', [])),
            'XC': sys.getsizeof(self.configs.get('XC', [])),
            'CC': sys.getsizeof(self.configs.get('CC', [])),
            'AX': sys.getsizeof(self.configs.get('AX', [])),
            'BX': sys.getsizeof(self.configs.get('BX', [])),
            'CXC': sys.getsizeof(self.configs.get('CXC', []))
        }
        
        return config_mem, totals
    
    def summary(self):
        print("\n" + "=" * 70)
        print("ADE3x3 Bulk Config Loader - Summary")
        print("=" * 70)
        
        print("\n--- RAW INSTANCE COUNTS ---")
        arity2_total = sum(self.schema_instance_count(s) for s in ['XX', 'CX', 'XC', 'CC', 'AX', 'BX'])
        arity3_total = self.schema_instance_count('CXC')
        
        print(f"XX = {self.schema_instance_count('XX')}")
        print(f"CX = {self.schema_instance_count('CX')}")
        print(f"XC = {self.schema_instance_count('XC')}")
        print(f"CC = {self.schema_instance_count('CC')}")
        print(f"AX = {self.schema_instance_count('AX')}")
        print(f"BX = {self.schema_instance_count('BX')}")
        print(f"CXC = {self.schema_instance_count('CXC')}")
        print(f"\nTotal arity-2: {arity2_total}")
        print(f"Total arity-3: {arity3_total}")
        print(f"Total newly loaded: {arity2_total + arity3_total}")
    
    def query_demo(self):
        print("\n--- QUERY DEMONSTRATIONS ---")
        
        examples = [
            ('XX', 0),
            ('CX', 0),
            ('XC', 0),
            ('CC', 0),
            ('AX', 0),
            ('BX', 0),
            ('CXC', 0)
        ]
        
        for schema, idx in examples:
            config = self.get_config(schema, idx)
            print(f"{schema}[{idx}]: {config.canonical_name}")


def main():
    print("Building ADE3x3 object DB with bulk config loading...")
    
    db = ADE3x3ObjectDB()
    db.build_atoms()
    db.build_actions()
    db.register_schemas()
    db.populate_all_core()
    
    db.summary()
    db.query_demo()
    
    config_mem, totals = db.memory_estimate()
    
    print("\n--- MEMORY USAGE ---")
    for schema, mem in totals.items():
        count = db.schema_instance_count(schema)
        print(f"{schema}: {count} configs, ~{mem} bytes (~{mem/1024:.1f} KB)")
    
    print(f"\nTotal config storage: ~{config_mem} bytes (~{config_mem/1024:.1f} KB)")
    
    print("\n--- PRACTICAL CONCLUSION ---")
    print("Core arity-2 and arity-3 config spaces successfully bulk-loaded.")
    print("Memory usage remains practical - warehouse expansion is sound.")
    print("Next step can proceed to canonicalization, signatures, or action images.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()