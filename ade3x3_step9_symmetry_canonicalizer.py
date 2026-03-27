import sys
import ade3x3_step1 as s1
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

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

@dataclass
class ConfigRecord:
    config_id: int
    schema_name: str
    slot_values: Tuple
    canonical_name: str

class ADE3x3DBWithSymmetry:
    def __init__(self):
        self.atoms_by_species = {'A': {}, 'B': {}, 'C': {}, 'X': {}}
        self.actions: List[ActionRecord] = []
        self.schemas: Dict[str, SchemaRecord] = {}
        self.configs: Dict[str, List[ConfigRecord]] = {}
        self.config_count: Dict[str, int] = {}
        
        self.canonical_id: Dict[str, Dict[int, int]] = {}
        self.orbit_id_of_config: Dict[str, Dict[int, int]] = {}
        self.orbit_members: Dict[str, Dict[int, List[int]]] = {}
        self.orbit_size: Dict[str, Dict[int, int]] = {}
        self.orbit_stabilizer: Dict[str, Dict[int, int]] = {}
        
        self._build_atoms()
        self._build_actions()
        self._register_schemas()
        self._populate_configs()
    
    def _build_atoms(self):
        for r in range(3):
            for s in range(3):
                local_idx = 3 * r + s
                self.atoms_by_species['A'][local_idx] = f"A[{r},{s}]"
        
        for t in range(3):
            for u in range(3):
                local_idx = 3 * t + u
                self.atoms_by_species['B'][local_idx] = f"B[{t},{u}]"
        
        for r in range(3):
            for u in range(3):
                local_idx = 3 * r + u
                self.atoms_by_species['C'][local_idx] = f"C[{r},{u}]"
        
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    for u_ in range(3):
                        local_idx = 9 * (3 * r + s) + (3 * t + u_)
                        self.atoms_by_species['X'][local_idx] = f"X[{r},{s}|{t},{u_}]"
    
    def _build_actions(self):
        s3 = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for pi_rA in s3:
            for pi_shared in s3:
                for pi_cB in s3:
                    rec = ActionRecord(len(self.actions), pi_rA, pi_shared, pi_cB)
                    self.actions.append(rec)
    
    def _register_schemas(self):
        self.schemas['XX'] = SchemaRecord('XX', ['X', 'X'], 2)
        self.schemas['CX'] = SchemaRecord('CX', ['C', 'X'], 2)
        self.schemas['XC'] = SchemaRecord('XC', ['X', 'C'], 2)
        self.schemas['CC'] = SchemaRecord('CC', ['C', 'C'], 2)
        self.schemas['AX'] = SchemaRecord('AX', ['A', 'X'], 2)
        self.schemas['BX'] = SchemaRecord('BX', ['B', 'X'], 2)
        self.schemas['CXC'] = SchemaRecord('CXC', ['C', 'X', 'C'], 3)
    
    def _populate_configs(self):
        for x1 in range(81):
            for x2 in range(81):
                name = f"XX[{self.atoms_by_species['X'][x1]},{self.atoms_by_species['X'][x2]}]"
                self.configs.setdefault('XX', []).append(ConfigRecord(0, 'XX', (x1, x2), name))
        self.config_count['XX'] = len(self.configs['XX'])
        
        for c in range(9):
            for x in range(81):
                name = f"CX[{self.atoms_by_species['C'][c]},{self.atoms_by_species['X'][x]}]"
                self.configs.setdefault('CX', []).append(ConfigRecord(0, 'CX', (c, x), name))
        self.config_count['CX'] = len(self.configs['CX'])
        
        for x in range(81):
            for c in range(9):
                name = f"XC[{self.atoms_by_species['X'][x]},{self.atoms_by_species['C'][c]}]"
                self.configs.setdefault('XC', []).append(ConfigRecord(0, 'XC', (x, c), name))
        self.config_count['XC'] = len(self.configs['XC'])
        
        for c1 in range(9):
            for c2 in range(9):
                name = f"CC[{self.atoms_by_species['C'][c1]},{self.atoms_by_species['C'][c2]}]"
                self.configs.setdefault('CC', []).append(ConfigRecord(0, 'CC', (c1, c2), name))
        self.config_count['CC'] = len(self.configs['CC'])
        
        for a in range(9):
            for x in range(81):
                name = f"AX[{self.atoms_by_species['A'][a]},{self.atoms_by_species['X'][x]}]"
                self.configs.setdefault('AX', []).append(ConfigRecord(0, 'AX', (a, x), name))
        self.config_count['AX'] = len(self.configs['AX'])
        
        for b in range(9):
            for x in range(81):
                name = f"BX[{self.atoms_by_species['B'][b]},{self.atoms_by_species['X'][x]}]"
                self.configs.setdefault('BX', []).append(ConfigRecord(0, 'BX', (b, x), name))
        self.config_count['BX'] = len(self.configs['BX'])
        
        for c1 in range(9):
            for x in range(81):
                for c2 in range(9):
                    name = f"CXC[{self.atoms_by_species['C'][c1]},{self.atoms_by_species['X'][x]},{self.atoms_by_species['C'][c2]}]"
                    self.configs.setdefault('CXC', []).append(ConfigRecord(0, 'CXC', (c1, x, c2), name))
        self.config_count['CXC'] = len(self.configs['CXC'])
        
        for schema in self.configs:
            for i, cfg in enumerate(self.configs[schema]):
                cfg.config_id = i
    
    def _act_on_a(self, idx, act):
        r, s = divmod(idx, 3)
        return 3 * act.pi_rA[r] + act.pi_shared[s]
    
    def _act_on_b(self, idx, act):
        return 0
    
    def _act_on_c(self, idx, act):
        r, u = divmod(idx, 3)
        return 3 * act.pi_rA[r] + act.pi_cB[u]
    
    def _act_on_x(self, idx, act):
        r, s = divmod(idx // 9, 3)
        t, u = divmod(idx % 9, 3)
        return 9 * (3 * act.pi_rA[r] + act.pi_shared[s]) + (3 * act.pi_shared[t] + act.pi_cB[u])
    
    def _transform_slots(self, slots, species_list, act):
        new_slots = []
        for slot, spec in zip(slots, species_list):
            if spec == 'A':
                new_slots.append(self._act_on_a(slot, act))
            elif spec == 'B':
                new_slots.append(slot)
            elif spec == 'C':
                new_slots.append(self._act_on_c(slot, act))
            elif spec == 'X':
                new_slots.append(self._act_on_x(slot, act))
            else:
                new_slots.append(slot)
        return tuple(new_slots)
    
    def compute_canonical_representatives(self):
        print("Computing canonical representatives...")
        
        for schema_name, configs in self.configs.items():
            species_list = self.schemas[schema_name].species_list
            
            slot_to_config = {}
            for cfg in configs:
                slot_to_config[cfg.slot_values] = cfg.config_id
            
            canonical_ids = {}
            
            for cfg in configs:
                min_slot = cfg.slot_values
                for act in self.actions:
                    new_slot = self._transform_slots(cfg.slot_values, species_list, act)
                    if new_slot < min_slot:
                        min_slot = new_slot
                canonical_ids[cfg.config_id] = slot_to_config[min_slot]
            
            self.canonical_id[schema_name] = canonical_ids
            
            orbit_groups = defaultdict(list)
            for cid, can in canonical_ids.items():
                orbit_groups[can].append(cid)
            
            orbit_members = {}
            orbit_size = {}
            orbit_id_of = {}
            
            oid = 0
            for repr_cid, members in sorted(orbit_groups.items()):
                for m in members:
                    orbit_id_of[m] = oid
                orbit_members[oid] = members
                orbit_size[oid] = len(members)
                oid += 1
            
            self.orbit_id_of_config[schema_name] = orbit_id_of
            self.orbit_members[schema_name] = orbit_members
            self.orbit_size[schema_name] = orbit_size
    
    def compute_stabilizers(self):
        print("Computing stabilizer sizes...")
        
        for schema_name in self.configs:
            orbit_stab = {}
            species_list = self.schemas[schema_name].species_list
            
            for repr_cid, members in self.orbit_members[schema_name].items():
                stab_count = 0
                for act in self.actions:
                    cfg = self.configs[schema_name][repr_cid]
                    new_slot = self._transform_slots(cfg.slot_values, species_list, act)
                    if new_slot == cfg.slot_values:
                        stab_count += 1
                orbit_stab[repr_cid] = stab_count
            
            self.orbit_stabilizer[schema_name] = orbit_stab
    
    def get_config(self, schema, cid):
        return self.configs[schema][cid]
    
    def report(self):
        print("\n" + "=" * 70)
        print("ADE3x3 Symmetry Canonicalizer - Report")
        print("=" * 70)
        
        print("\n--- CONFIGURATION COUNTS ---")
        total = 0
        for schema, count in self.config_count.items():
            print(f"{schema}: {count} raw configs")
            total += count
        print(f"Total: {total}")
        
        print("\n--- ORBIT SUMMARIES ---")
        for schema in self.configs:
            num_orbits = len(self.orbit_members[schema])
            sizes = list(self.orbit_size[schema].values())
            print(f"{schema}: {self.config_count[schema]} raw -> {num_orbits} orbits")
            print(f"   avg={sum(sizes)/len(sizes):.1f}, min={min(sizes)}, max={max(sizes)}")
        
        print("\n--- CANONICAL EXAMPLES ---")
        for schema in ['CX', 'XX', 'CXC']:
            repr_cid = list(self.orbit_members[schema].keys())[0]
            cfg = self.get_config(schema, repr_cid)
            print(f"{schema} orbit repr: {cfg.canonical_name}")
        
        print("\n--- QUERY EXAMPLES ---")
        cfg = self.get_config('CX', 0)
        can = self.canonical_id['CX'][0]
        print(f"CX[0] = {cfg.canonical_name}")
        print(f"  canonical = {self.get_config('CX', can).canonical_name}")
        
        oid = self.orbit_id_of_config['CXC'][5000]
        print(f"CXC[5000] orbit_id = {oid}, size = {self.orbit_size['CXC'][oid]}")
        
        print("\n--- MEMORY (APPROX) ---")
        canon_total = sum(sum(sys.getsizeof(v) for v in d.values()) for d in self.canonical_id.values())
        print(f"Canonical ID table: ~{canon_total/1024:.1f} KB")
        print("Action images computed on-the-fly (not stored), orbits indexed.")


def main():
    print("Initializing DB with symmetry canonicalization...")
    db = ADE3x3DBWithSymmetry()
    db.compute_canonical_representatives()
    db.compute_stabilizers()
    db.report()
    print("\n--- COMPLETE ---")


if __name__ == "__main__":
    main()