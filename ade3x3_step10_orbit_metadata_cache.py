import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

@dataclass
class OrbitMetadataRecord:
    schema_name: str
    orbit_id: int
    repr_config_id: int
    repr_tuple: Tuple
    repr_name: str
    orbit_size: int
    stabilizer_size: int
    signature_key: Tuple
    projection_cx: Optional[Tuple] = None
    projection_xc: Optional[Tuple] = None
    projection_cc: Optional[Tuple] = None
    projection_cx_orbit: Optional[int] = None
    projection_xc_orbit: Optional[int] = None
    projection_cc_orbit: Optional[int] = None


class ADE3x3OrbitMetadataDB:
    def __init__(self):
        self.atoms_by_species = {'A': {}, 'B': {}, 'C': {}, 'X': {}}
        self.actions = []
        self.schemas = {}
        self.configs = {}
        self.config_count = {}
        
        self.canonical_id = {}
        self.orbit_id_of_config = {}
        self.orbit_members = {}
        self.orbit_size = {}
        self.orbit_stabilizer = {}
        
        self.orbit_metadata: Dict[str, Dict[int, OrbitMetadataRecord]] = {}
        
        self._init_atoms()
        self._init_actions()
        self._register_schemas()
        self._populate_configs()
        self._compute_canonical_and_orbits()
        self._compute_stabilizers()
        self._compute_orbit_metadata()
    
    def _init_atoms(self):
        for r in range(3):
            for s in range(3):
                self.atoms_by_species['A'][3*r+s] = f"A[{r},{s}]"
        for t in range(3):
            for u in range(3):
                self.atoms_by_species['B'][3*t+u] = f"B[{t},{u}]"
        for r in range(3):
            for u in range(3):
                self.atoms_by_species['C'][3*r+u] = f"C[{r},{u}]"
        for r in range(3):
            for s in range(3):
                for t in range(3):
                    for u_ in range(3):
                        self.atoms_by_species['X'][9*(3*r+s)+(3*t+u_)] = f"X[{r},{s}|{t},{u_}]"
    
    def _init_actions(self):
        s3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        for pi_rA in s3:
            for pi_shared in s3:
                for pi_cB in s3:
                    self.actions.append((pi_rA, pi_shared, pi_cB))
    
    def _register_schemas(self):
        self.schemas = {
            'XX': ['X', 'X'], 'CX': ['C', 'X'], 'XC': ['X', 'C'],
            'CC': ['C', 'C'], 'AX': ['A', 'X'], 'BX': ['B', 'X'],
            'CXC': ['C', 'X', 'C']
        }
    
    def _populate_configs(self):
        for x1 in range(81):
            for x2 in range(81):
                self.configs.setdefault('XX', []).append(((x1, x2), f"XX[{self.atoms_by_species['X'][x1]},{self.atoms_by_species['X'][x2]}]"))
        self.config_count['XX'] = len(self.configs['XX'])
        
        for c in range(9):
            for x in range(81):
                self.configs.setdefault('CX', []).append(((c, x), f"CX[{self.atoms_by_species['C'][c]},{self.atoms_by_species['X'][x]}]"))
        self.config_count['CX'] = len(self.configs['CX'])
        
        for x in range(81):
            for c in range(9):
                self.configs.setdefault('XC', []).append(((x, c), f"XC[{self.atoms_by_species['X'][x]},{self.atoms_by_species['C'][c]}]"))
        self.config_count['XC'] = len(self.configs['XC'])
        
        for c1 in range(9):
            for c2 in range(9):
                self.configs.setdefault('CC', []).append(((c1, c2), f"CC[{self.atoms_by_species['C'][c1]},{self.atoms_by_species['C'][c2]}]"))
        self.config_count['CC'] = len(self.configs['CC'])
        
        for a in range(9):
            for x in range(81):
                self.configs.setdefault('AX', []).append(((a, x), f"AX[{self.atoms_by_species['A'][a]},{self.atoms_by_species['X'][x]}]"))
        self.config_count['AX'] = len(self.configs['AX'])
        
        for b in range(9):
            for x in range(81):
                self.configs.setdefault('BX', []).append(((b, x), f"BX[{self.atoms_by_species['B'][b]},{self.atoms_by_species['X'][x]}]"))
        self.config_count['BX'] = len(self.configs['BX'])
        
        for c1 in range(9):
            for x in range(81):
                for c2 in range(9):
                    self.configs.setdefault('CXC', []).append(((c1, x, c2), f"CXC[{self.atoms_by_species['C'][c1]},{self.atoms_by_species['X'][x]},{self.atoms_by_species['C'][c2]}]"))
        self.config_count['CXC'] = len(self.configs['CXC'])
    
    def _transform(self, slots, species_list, act):
        pi_rA, pi_shared, pi_cB = act
        new_slots = []
        for slot, spec in zip(slots, species_list):
            if spec == 'A':
                r, s = divmod(slot, 3)
                new_slots.append(3*pi_rA[r] + pi_shared[s])
            elif spec == 'B':
                new_slots.append(slot)
            elif spec == 'C':
                r, u = divmod(slot, 3)
                new_slots.append(3*pi_rA[r] + pi_cB[u])
            elif spec == 'X':
                r, s = divmod(slot // 9, 3)
                t, u = divmod(slot % 9, 3)
                new_a = 3*pi_rA[r] + pi_shared[s]
                new_b = 3*pi_shared[t] + pi_cB[u]
                new_slots.append(9*new_a + new_b)
            else:
                new_slots.append(slot)
        return tuple(new_slots)
    
    def _compute_canonical_and_orbits(self):
        for schema, configs in self.configs.items():
            species_list = self.schemas[schema]
            slot_to_idx = {t[0]: i for i, t in enumerate(configs)}
            
            can_id = {}
            for cfg_idx, (slots, _) in enumerate(configs):
                min_slot = slots
                for act in self.actions:
                    new_slot = self._transform(slots, species_list, act)
                    if new_slot < min_slot:
                        min_slot = new_slot
                can_id[cfg_idx] = slot_to_idx[min_slot]
            
            self.canonical_id[schema] = can_id
            
            groups = defaultdict(list)
            for cid, can in can_id.items():
                groups[can].append(cid)
            
            orbit_members = {}
            orbit_id_of = {}
            for oid, (repr_cid, members) in enumerate(sorted(groups.items())):
                for m in members:
                    orbit_id_of[m] = oid
                orbit_members[oid] = members
            
            self.orbit_id_of_config[schema] = orbit_id_of
            self.orbit_members[schema] = orbit_members
    
    def _compute_stabilizers(self):
        for schema, members in self.orbit_members.items():
            species_list = self.schemas[schema]
            stabs = {}
            for repr_cid, orbid in members.items():
                cfg_slots = self.configs[schema][repr_cid][0]
                stab_count = sum(1 for act in self.actions if self._transform(cfg_slots, species_list, act) == cfg_slots)
                stabs[repr_cid] = stab_count
            self.orbit_stabilizer[schema] = stabs
    
    def _signature_XX(self, slots):
        x1, x2 = slots
        r1, s1 = divmod(x1 // 9, 3)
        t1, u1 = divmod(x1 % 9, 3)
        r2, s2 = divmod(x2 // 9, 3)
        t2, u2 = divmod(x2 % 9, 3)
        live1 = (s1 == t1)
        live2 = (s2 == t2)
        return (live1, live2, r1==r2, s1==s2, t1==t2, u1==u2, r1==r2 and s1==s2, t1==t2 and u1==u2)
    
    def _signature_CX(self, slots):
        c, x = slots
        r_u = divmod(c, 3)
        r, s = divmod(x // 9, 3)
        t, u = divmod(x % 9, 3)
        live = (s == t)
        return (live, live and c == 3*r+u, r_u[0]==r, r_u[1]==u)
    
    def _signature_XC(self, slots):
        x, c = slots
        r, s = divmod(x // 9, 3)
        t, u = divmod(x % 9, 3)
        r_u = divmod(c, 3)
        live = (s == t)
        return (live, live and c == 3*r+u, r==r_u[0], u==r_u[1])
    
    def _signature_CC(self, slots):
        c1, c2 = slots
        r1, u1 = divmod(c1, 3)
        r2, u2 = divmod(c2, 3)
        return (c1==c2, r1==r2, u1==u2)
    
    def _signature_AX(self, slots):
        a, x = slots
        r_a, s_a = divmod(a, 3)
        r, s = divmod(x // 9, 3)
        t, u = divmod(x % 9, 3)
        live = (s == t)
        return (r_a==r, s_a==s, live)
    
    def _signature_BX(self, slots):
        b, x = slots
        t_b, u_b = divmod(b, 3)
        r, s = divmod(x // 9, 3)
        t, u = divmod(x % 9, 3)
        live = (s == t)
        return (t_b==t, u_b==u, live)
    
    def _signature_CXC(self, slots):
        c1, x, c2 = slots
        r1, u1 = divmod(c1, 3)
        r, s = divmod(x // 9, 3)
        t, u = divmod(x % 9, 3)
        r2, u2 = divmod(c2, 3)
        live = (s == t)
        return (
            live, c1==c2,
            live and c1 == 3*r+u, live and c2 == 3*r+u,
            (r1, r, r2), (u1, u, u2)
        )
    
    def _compute_orbit_metadata(self):
        for schema in self.configs:
            meta = {}
            species_list = self.schemas[schema]
            
            sig_func = getattr(self, f'_signature_{schema}', None)
            
            for repr_cid, members in self.orbit_members[schema].items():
                slots, name = self.configs[schema][repr_cid]
                orbit_size = len(members)
                stab_size = self.orbit_stabilizer[schema][repr_cid]
                sig = sig_func(slots) if sig_func else ()
                
                entry = OrbitMetadataRecord(
                    schema_name=schema,
                    orbit_id=0,
                    repr_config_id=repr_cid,
                    repr_tuple=slots,
                    repr_name=name,
                    orbit_size=orbit_size,
                    stabilizer_size=stab_size,
                    signature_key=sig
                )
                meta[repr_cid] = entry
            
            self.orbit_metadata[schema] = meta
    
    def get_orbit_metadata(self, schema, orbit_id):
        for cid, meta in self.orbit_metadata[schema].items():
            if meta.orbit_id == orbit_id:
                return meta
        return None
    
    def get_signature_key(self, schema, orbit_id):
        meta = self.get_orbit_metadata(schema, orbit_id)
        return meta.signature_key if meta else None
    
    def report(self):
        print("\n" + "=" * 70)
        print("ADE3x3 Orbit Metadata Cache - Report")
        print("=" * 70)
        
        print("\n--- ORBIT METADATA SUMMARY ---")
        for schema in self.configs:
            num_orbits = len(self.orbit_members[schema])
            sizes = [m.orbit_size for m in self.orbit_metadata[schema].values()]
            stabs = [m.stabilizer_size for m in self.orbit_metadata[schema].values()]
            print(f"{schema}: {self.config_count[schema]} raw -> {num_orbits} orbits")
            print(f"   orbit_size: avg={sum(sizes)/len(sizes):.1f}, min={min(sizes)}, max={max(sizes)}")
            print(f"   stabilizer: min={min(stabs)}, max={max(stabs)}")
            print(f"   orbit*stab check: {sizes[0]*stabs[0]}")
        
        print("\n--- SIGNATURE SUMMARY ---")
        for schema in self.configs:
            sig_keys = set(m.signature_key for m in self.orbit_metadata[schema].values())
            print(f"{schema}: {len(self.orbit_members[schema])} orbits, {len(sig_keys)} distinct signatures")
        
        print("\n--- PROJECTION SUMMARY (CXC) ---")
        for meta in list(self.orbit_metadata['CXC'].values())[:3]:
            c1, x, c2 = meta.repr_tuple
            cx_name = f"CX[{self.atoms_by_species['C'][c1]},{self.atoms_by_species['X'][x]}]"
            xc_name = f"XC[{self.atoms_by_species['X'][x]},{self.atoms_by_species['C'][c2]}]"
            cc_name = f"CC[{self.atoms_by_species['C'][c1]},{self.atoms_by_species['C'][c2]}]"
            print(f"  CXC rep: {cx_name} | {xc_name} | {cc_name}")
        
        print("\n--- QUERY DEMOS ---")
        print("XX orbit metadata:")
        meta = list(self.orbit_metadata['XX'].values())[0]
        print(f"  repr: {meta.repr_name}, orbit_size={meta.orbit_size}, stab={meta.stabilizer_size}, sig={meta.signature_key}")
        
        print("CX orbit metadata:")
        meta = list(self.orbit_metadata['CX'].values())[0]
        print(f"  repr: {meta.repr_name}, orbit_size={meta.orbit_size}, stab={meta.stabilizer_size}, sig={meta.signature_key}")
        
        print("CXC orbit metadata:")
        meta = list(self.orbit_metadata['CXC'].values())[0]
        print(f"  repr: {meta.repr_name}, orbit_size={meta.orbit_size}, stab={meta.stabilizer_size}, sig={meta.signature_key}")
        
        print("\n--- MEMORY ---")
        meta_mem = sum(sys.getsizeof(m) for schema in self.orbit_metadata.values() for m in schema.values())
        print(f"Orbit metadata cache: ~{meta_mem/1024:.1f} KB")


def main():
    print("Building orbit metadata cache...")
    db = ADE3x3OrbitMetadataDB()
    db.report()
    print("\n--- COMPLETE ---")
    print("Warehouse now has semantic orbit indexing for all core schemas.")


if __name__ == "__main__":
    main()