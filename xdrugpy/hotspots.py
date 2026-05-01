from __future__ import annotations

import os.path
import tempfile
from types import SimpleNamespace
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Literal, List, Dict, Tuple
from itertools import combinations
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
from scipy.spatial import distance_matrix, distance, cKDTree

from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from strenum import StrEnum
import networkx as nx
import pyKVFinder

from .utils import (
    new_command,
    Selection,
    plot_hca_base,
)

from pymol import cmd as pm


matplotlib.use("Qt5Agg")


@dataclass
class Cluster:
    selection: str
    coords: Any = field(repr=False, hash=False)
    ST: int


@dataclass
class ECluster:
    selection: str
    probe_type: str
    coords: Any = field(repr=False, hash=False)
    idx: int
    ST: int


# @lru_cache(1024)
def get_coords(sel, state=1):
    return pm.get_coords(sel, state)


def get_clusters():
    clusters = []
    eclusters = []
    for obj in pm.get_object_list():
        if obj.startswith(f"crosscluster."):
            _, _, s, _ = obj.split(".", maxsplit=4)
            pm.remove(f"%{obj} & elem H")
            coords = get_coords(obj)
            clusters.append(
                Cluster(
                    selection=obj,
                    coords=coords,
                    ST=int(s),
                )
            )
        elif obj.startswith("consensus."):
            _, _, s = obj.split(".", maxsplit=3)
            pm.remove(f"%{obj} & elem H")
            coords = get_coords(obj)
            clusters.append(
                Cluster(
                    selection=obj,
                    coords=coords,
                    ST=int(s),
                )
            )
        elif obj.startswith("clust."):
            _, idx, s, probe_type = obj.split(".", maxsplit=4)
            coords = get_coords(obj)
            eclusters.append(
                ECluster(
                    selection=obj,
                    probe_type=probe_type,
                    coords=coords,
                    idx=int(idx),
                    ST=int(s),
                )
            )
    return clusters, eclusters


def set_properties(obj, obj_name, properties):
    for prop, value in properties.items():
        pm.set_property(prop, value, obj_name)
        pm.set_atom_property(prop, value, obj_name)
        setattr(obj, prop, value)


def find_occupied_pockets(
        group: str,
        pocket_residues: Dict[str, List[Any]],
        clusters: List[Cluster],
) -> List[Tuple[str, List[Cluster]]]:
    pockets = {}
    for key, pocket in pocket_residues.items():
        p_sele = ''
        for resi, chain, _ in pocket:
            p_sele += f'(chain {chain} & resi {resi}) | '
        p_sele += 'none'
        p_sele = f'{group}.protein & ({p_sele})'

        levels = []
        for radius in [4, 8]:
            hs_sele = f"{group}.CS.* near_to {radius} of ({p_sele})"
            sele_cnt = 0
            while True:
                new_sele_cnt = pm.count_atoms(hs_sele)
                if new_sele_cnt != sele_cnt:
                    sele_cnt = new_sele_cnt
                else:
                    break
                levels.append(hs_sele)
                hs_sele = f"{group}.CS.* within {radius} of ({hs_sele})"
        
        for hs_sele in levels:
            hs_objs = pm.get_object_list(hs_sele)

            if not hs_objs:
                continue

            pocket_clusters = [c for c in clusters if c.selection in hs_objs]
            hs_sele = ' | '.join([c.selection for c in pocket_clusters])
            pockets[hs_sele] = pocket_clusters
    return pockets


def find_pykvf_pockets(protein):
    with tempfile.TemporaryDirectory() as tempdir:
        protein_pdb = f"{tempdir}/protein.pdb"
        pm.save(protein_pdb, selection=protein)
        atomic = pyKVFinder.read_pdb(protein_pdb)
        
    vertices = pyKVFinder.get_vertices(atomic)
    _, cavities = pyKVFinder.detect(atomic, vertices)
    residues = pyKVFinder.constitutional(cavities, atomic, vertices)
    return residues

def process_clusters(group, clusters):
    for idx, cs in enumerate(clusters):
        new_name = f"{group}.CS.{idx:02}"
        pm.set_name(cs.selection, new_name)
        cs.selection = new_name
        pm.group(group, new_name)
        set_properties(
            cs,
            new_name,
            {
                "Type": "CS",
                "Group": group,
                "Selection": new_name,
                "ST": cs.ST,
            },
        )
    pm.delete("consensus.*")
    pm.delete("crosscluster.*")


def process_eclusters(group, eclusters):
    for acs in eclusters:
        new_name = f"{group}.ACS.{acs.probe_type}.{acs.idx:02}"
        pm.set_name(acs.selection, new_name)
        acs.selection = new_name
        pm.group(group, new_name)

        coords = pm.get_coordset(new_name)
        md = distance_matrix(coords, coords).max()

        set_properties(
            acs,
            new_name,
            {
                "Type": "ACS",
                "Group": group,
                "Selection": new_name,
                "Class": acs.probe_type,
                "ST": acs.ST,
                "MD": round(md, 2),
            },
        )
    pm.delete("clust.*")


@dataclass
class Hotspot:
    group: str
    selection: str
    clusters: List[Cluster] = field(repr=False, compare=False, hash=False)
    cluster_list: str = field(repr=False, compare=True)

    klass: Optional[Literal["D", "DS", "DL", "B", "BS", "BL"]]
    ST: int
    S0: int
    S1: int
    SZ: int
    CD: float
    CD0: float
    CD16: float
    CD13: float
    MD: float
    length: int
    nComponents: int
    type: Literal["HS"] =  field(default="HS", repr=False)
    
    def copy_into_properties(self):
        d = asdict(self)
        del d['clusters'], d['type']
        set_properties(
            SimpleNamespace(),
            self.selection,
            {
                **d,
                'Group': self.group,
                'Type': 'HS',
                'Class': self.klass,
                'Length': self.length,
                'cluster_list': self.cluster_list,
            }
        )
    
    @staticmethod
    def from_cluster_selections(
        selections: List[str],
        cd_to_anchor: bool=True,
        max_collisions: float=0.10
    ) -> Hotspot:
        clusters = []
        objs = pm.get_object_list(' | '.join(selections))
        for obj in objs:
            group = obj.split('.')[0]
            assert obj.startswith(f"{group}.CS.")
            clu = Cluster(
                obj,
                pm.get_coordset(obj),
                count_molecules(obj)
            )
            pm.group(group, obj, action='add')
            clusters.append(clu)
        return Hotspot.from_clusters(group, clusters, cd_to_anchor=cd_to_anchor, max_collisions=max_collisions)
        
    @staticmethod
    def from_clusters(group: str, clusters: List[Cluster], cd_to_anchor: bool=True, max_collisions: float=0.10) -> Hotspot:
        coms = [pm.centerofmass(c.selection) for c in clusters]
        cd = [distance.euclidean(coms[0], com) for com in coms]
        cd0 = np.max(cd) if len(cd) > 0 else 0.0

        if cd_to_anchor:

            cd16 = [] 
            coms16 = [pm.centerofmass(c.selection) for c in clusters if c.ST >= 16]
            for com in coms:
                min_d = 0.0
                for com16 in coms16:
                    d = distance.euclidean(com, com16)
                    if min_d == 0.0 or d < min_d:
                        min_d = d
                cd16.append(min_d)
        
            cd13 = []
            coms13 = [pm.centerofmass(c.selection) for c in clusters if c.ST >= 13]
            for com in coms:
                min_d = 0.0
                for com13 in coms13:
                    d = distance.euclidean(com, com13)
                    if min_d == 0.0 or d < min_d:
                        min_d = d
                cd13.append(min_d)
            
            cd16 = np.max(cd16) if len(cd16) > 0 else 0.0
            cd13 = np.max(cd13) if cd16 == 0.0 and len(cd13) > 0 else 0.0
        else:
            cd16 = 0.0
            cd13 = 0.0

        coords = np.concatenate([c.coords for c in clusters])
        max_dist = distance_matrix(coords, coords).max()

        selection = " ".join(c.selection for c in clusters)
        hs = Hotspot(
            group=group,
            selection=selection,
            clusters=clusters,
            cluster_list=selection,
            klass=None,
            ST=sum(c.ST for c in clusters),
            S0=clusters[0].ST,
            S1=clusters[1].ST if len(clusters) >= 2 else 0,
            SZ=clusters[-1].ST,
            CD0=cd0,
            CD16=cd16,
            CD13=cd13,
            CD=0.0,
            MD=max_dist,
            length=len(clusters),
            nComponents=-1,
        )   
        
        s0 = hs.S0
        if cd_to_anchor:
            cd = (
                cd16 if cd16 > 0 else
                cd13 if cd13 > 0 else
                cd0
            )
        else:
            cd = cd0
        hs.CD = cd
        md = hs.MD

        if s0 >= 16 and cd < 8 and md >= 10:
            hs.klass = "D"
        if s0 >= 16 and cd < 8 and 7 <= md < 10:
            hs.klass = "DS"
        if 13 <= s0 < 16 and cd < 8 and md >= 10:
            hs.klass = "B"
        if 13 <= s0 < 16 and cd < 8 and 7 <= md < 10:
            hs.klass = "BS"
        if 13 <= s0 < 16 and cd >= 8 and md >= 10:
            hs.klass = "BL" 
        if s0 >= 16 and cd >= 8 and md >= 10:
            hs.klass = "DL"
        
        if hs.klass:
            hs.nComponents = Hotspot.make_graph(group, clusters, max_collisions=max_collisions)
            if hs.nComponents > 1:
                hs.klass = None
        return hs
    
    @staticmethod
    def find_hotspots(
        group: str,
        pocket_residues: Dict[str, Any],
        clusters: List[Cluster],
        cd_to_anchor: bool,
        combinatory_search: bool,
        allow_nested: bool,
        max_collisions: float=0.10
    ) -> List[Hotspot]:
        
        # filter out weak clusters
        ix_ignore = next((i for i, c in enumerate(clusters) if c.ST<5), -1)
        clusters = clusters[:ix_ignore]

        # identify hotspots from pockets and consensus sites
        spots = []
        pockets = find_occupied_pockets(group, pocket_residues, clusters)
        for hs_sele, pocket_clusters in pockets.items():
            if pocket_clusters:
                hs = Hotspot.from_clusters(group, pocket_clusters, cd_to_anchor=cd_to_anchor, max_collisions=max_collisions)
                if hs.klass:
                    hs.selection = hs_sele
                    spots.append(hs)
        
        # identify hotspots from combinations of consensus sites
        if combinatory_search:
            for r in range(1, 4):
                for comb in combinations(clusters, r):
                    comb = list(comb)
                    hs = Hotspot.from_clusters(group, comb, cd_to_anchor=cd_to_anchor, max_collisions=max_collisions)
                    if hs.klass:
                        spots.append(hs)
        
        # remove identical repeated hotspots
        repeated = set()
        for ix1, hs1 in enumerate(spots):
            for ix2, hs2 in enumerate(spots):
                if ix1 != ix2:
                    if ix1 < ix2:
                        continue
                    if len(hs1.clusters) == len(hs2.clusters):
                        if all(c1 == c2 for c1, c2 in zip(hs1.clusters, hs2.clusters)):
                            repeated.add(hs2.selection)
        
        for hs in spots.copy():
            if hs.selection in repeated:
                spots.remove(hs)

        # remove hotspot objects when they totally fit inside another (and are of the same class)
        if not allow_nested:
            nested = set()
            for ix1, hs1 in enumerate(spots):
                for ix2, hs2 in enumerate(spots):
                    if ix1 == ix2:
                        continue
                    if hs1.klass != hs2.klass:
                        continue
                    if get_fo(hs1.selection, hs2.selection) == 1:
                        nested.add(hs1.selection)
                        continue

            for hs in spots.copy():
                if hs.selection in nested:
                    spots.remove(hs)
                

        # rename and renumber hotspots
        spots = sorted(spots, key=lambda hs: (-hs.ST, -hs.S0, -hs.S1, -hs.SZ))
        spots = sorted(spots, key=lambda hs: ["D", "DS", "DL", "B", "BS", "BL"].index(hs.klass))
        spots = list(spots)

        ix_class = 0
        last_class = None
        for hs in spots.copy():
            if hs.klass != last_class:
                last_class = hs.klass
                ix_class = 0
            new_name = f"{group}.{hs.klass}.{ix_class:02}"
            pm.create(new_name, hs.selection)
            hs.selection = new_name
            ix_class += 1
        
        for hs in spots:
            hs.copy_into_properties()
        
        return spots
    
    def show(self, plot_graph: bool=False):
        group = self.group
        base_name = f"{group}.diagnose"
        pm.delete(f'{base_name}*')

        # create surface object
        surf_sele = f"{group}.protein near_to 5 of ({self.cluster_list})"
        surf_name = f"{base_name}.surf"
        pm.create(surf_name, surf_sele)
        pm.hide(selection=surf_name)
        pm.show(representation="surface", selection=surf_name)
        pm.set("transparency", 0.4, surf_name)

        for ix, clu in enumerate(self.clusters):
            clu = self.clusters[ix]
            S = clu.ST
            label_ps = pm.get_unused_name(f'{base_name}.label_ps_')
            pm.pseudoatom(
                label_ps,
                selection=clu.selection,
                label=f"{clu.selection}\nS={S}"
            )
            pm.set("float_label", 0, label_ps)

            if ix != 0:
                measure_name = pm.get_unused_name(f'{group}.diagnose_dist_')
                pm.distance(
                    measure_name,
                    self.clusters[0].selection,
                    self.clusters[ix].selection,
                    mode=4
                )
                pm.group(group, base_name, action='add')
            
            if plot_graph:
                for ix1, clu1 in enumerate(self.clusters):
                    for ix2, clu2 in enumerate(self.clusters):
                        if ix1 >= ix2:
                            continue
                        e = (clu1.selection, clu2.selection)
                        if e not in self.graph.edges:
                            measure_name = pm.get_unused_name(f'{group}.diagnose_graph_')
                            pm.distance(measure_name, e[0], e[1], mode=4)
                            pm.color('cyan', measure_name)

    def make_graph(group: str, clusters: List[Cluster], radius: float=1.7, samples: int=50, max_collisions: float=0.15) -> int:
        g = nx.Graph()
        # Adiciona todos os nomes de clusters como NÓS (essencial para contar isolados)
        g.add_nodes_from([c.selection for c in clusters])

        for ix1, c1 in enumerate(clusters):
            for ix2, c2 in enumerate(clusters):
                if ix1 >= ix2:
                    continue
                collisions = Hotspot.detect_collisions(group, c1.selection, c2.selection, radius, samples)
                # Lógica de Conectividade:
                # Um átomo em A está 'conectado' a B se ele tem pelo menos UM caminho livre para B
                # Se mais de 75% dos átomos de ambos os clusters conseguem se 'ver', estão no mesmo bolso
                if np.sum(collisions) < max_collisions*collisions.shape[0]*collisions.shape[1]:
                    g.add_edge(c1.selection, c2.selection)
                
        return nx.number_connected_components(g)

    @staticmethod
    @lru_cache
    def detect_collisions(group: str, clu_sele1: str, clu_sele2: str, radius: float, samples: int):

        list_a = get_coords(clu_sele1)
        list_b = get_coords(clu_sele2)
        
        prot_xyz = get_coords(f'{group}.protein')
        tree = cKDTree(prot_xyz)
        
        t = np.linspace(0, 1, samples)

        # Broadcasting para gerar pontos intermediários
        # Mágica avançada da IA
        A = list_a[:, np.newaxis, np.newaxis, :]
        B = list_b[np.newaxis, :, np.newaxis, :]
        T = t[np.newaxis, np.newaxis, :, np.newaxis]
        
        # Matriz (N, M, samples, 3)
        inter_points = A + T * (B - A)
        c1c2_xyz = inter_points.reshape(-1, 3)
        
        # Checa colisões com a proteína
        collisions = tree.query_ball_point(c1c2_xyz, r=radius)
        
        # True se o ponto i->j no frame k colide
        has_collision = np.array([
            len(c) > 0 for c in collisions
        ]).reshape(len(list_a), len(list_b), samples)
        
        # Um caminho entre o átomo i e o átomo j é CONSIDERADO LIVRE se não houver colisões em 'samples'
        has_collision = np.any(has_collision, axis=2) # Matriz (N, M)
        return has_collision
    
@new_command
def show_hs(selections: List[str],
            cd_to_anchor: bool = True,
            max_collisions: float = 0.15) -> Hotspot:
    hs = Hotspot.from_cluster_selections(
        selections,
        cd_to_anchor=cd_to_anchor,
        max_collisions=max_collisions
    )
    hs.show()
    print(hs)


class FTMapResults:
    pass

@new_command
def load_ftmap(
    filenames: List[Path] | Path,
    groups: List[str] | str = "",
    cd_to_anchor: bool = True,
    combinatory_search: bool = False,
    allow_nested: bool = False,
    max_collisions: float = 0.15,
):
    try:
        pm.set('defer_updates', 1)
        if isinstance(filenames, (str, Path)):
            filenames = [filenames]
            groups = [groups]
            single = True
        else:
            single = False
        rets = []
        if isinstance(groups, (tuple, list)):
            iter = zip(filenames, groups)
        else:
            iter = []
            for filename in filenames:
                iter.append((filename, os.path.splitext(os.path.basename(filename))[0]))
        for fnames, groups in iter:
            try:
                rets.append(_load_ftmap(fnames, groups, cd_to_anchor=cd_to_anchor, combinatory_search=combinatory_search, allow_nested=allow_nested, max_collisions=max_collisions))
            except:
                rets.append(_load_ftmap(fnames, groups, cd_to_anchor=cd_to_anchor, combinatory_search=combinatory_search, allow_nested=allow_nested, max_collisions=max_collisions))
        if single:
            return rets[0]
        else:
            return rets
    finally:
        pm.set('defer_updates', 0)


def _load_ftmap(
    filename: Path,
    group: str = "",
    cd_to_anchor: bool = True,
    combinatory_search: bool = False,
    allow_nested: bool = False,
    max_collisions: float = 0.15,
):
    """
    Load a FTMap PDB file and classify hotspot ensembles in accordance to
    Kozakov et al. (2015).
    https://doi.org/10.1021/acs.jmedchem.5b00586

    OPTIONS
        filename        mapping PDB file.
        group           optional group name.

    EXAMPLES
        load_ftmap ace_example.pdb
        load_ftmap ace_example.pdb, group=MyProtein
    """
    if not group:
        group = os.path.splitext(os.path.basename(filename))[0]
    group = pm.get_legal_name(group)
    
    pm.delete(f"%{group}")
    pm.load(str(filename), quiet=1, discrete=1)

    if objs := pm.get_object_list("*_protein"):
        assert len(objs) == 1
        protein = objs[0]
    elif pm.get_object_list("protein"):
        protein = "protein"
    pm.set_name(protein, f"{group}.protein")
    pm.group(group, f"{group}.protein")

    clusters, eclusters = get_clusters()
    process_clusters(group, clusters)
    process_eclusters(group, eclusters)
    pocket_residues = find_pykvf_pockets(f"{group}.protein")
    hotspots = Hotspot.find_hotspots(
        group,
        pocket_residues,
        clusters,
        cd_to_anchor=cd_to_anchor,
        combinatory_search=combinatory_search,
        allow_nested=allow_nested,
        max_collisions=max_collisions
    )

    pm.hide("everything", f"{group}.*")

    pm.show("cartoon", f"{group}.protein")
    pm.show("mesh", f"{group}.D* or {group}.B*")

    pm.show("spheres", f"{group}.ACS.*")
    pm.set("sphere_scale", 0.25, f"{group}.ACS.*")

    pm.color("red", f"{group}.D*")
    pm.color("salmon", f"{group}.B*")

    pm.color("red", f"{group}.ACS.acceptor.*")
    pm.color("blue", f"{group}.ACS.donor.*")
    pm.color("green", f"{group}.ACS.halogen.*")
    pm.color("orange", f"{group}.ACS.aromatic.*")
    pm.color("yellow", f"{group}.ACS.apolar.*")

    pm.show("line", f"{group}.CS.*")

    pm.disable(f"{group}.CS.*")

    pm.set("mesh_mode", 1)
    pm.orient("all")

    pm.order(f"{group}.CS.*", location="top")

    pm.order(f"{group}.BS.*", location="top")
    pm.order(f"{group}.BL.*", location="top")
    pm.order(f"{group}.B.*", location="top")
    pm.order(f"{group}.DS.*", location="top")
    pm.order(f"{group}.DL.*", location="top")
    pm.order(f"{group}.D.*", location="top")
    
    pm.order(f"{group}.protein", location="top")

    pm.group(group, f"{group}.*", 'add')
    pm.group(group, f"{group}.protein", 'add')

    pm.group(group, f"{group}.D")
    pm.group(group, f"{group}.B")
    pm.group(group, f"{group}.DS")
    pm.group(group, f"{group}.BS")
    pm.group(group, f"{group}.DL")
    pm.group(group, f"{group}.BL")

    pm.group(f"{group}.D", f"{group}.D.*")
    pm.group(f"{group}.B", f"{group}.B.*")
    pm.group(f"{group}.DS", f"{group}.DS.*")
    pm.group(f"{group}.BS", f"{group}.BS.*")
    pm.group(f"{group}.DL", f"{group}.DL.*")
    pm.group(f"{group}.BL", f"{group}.BL.*")

    pm.group(f"{group}.CS", f"{group}.CS.*")

    pm.group(group, f"{group}.ACS", 'add')
    pm.group(f"{group}.ACS", f"{group}.ACS.acceptor")
    pm.group(f"{group}.ACS", f"{group}.ACS.donor")
    pm.group(f"{group}.ACS", f"{group}.ACS.halogen")
    pm.group(f"{group}.ACS", f"{group}.ACS.aromatic")
    pm.group(f"{group}.ACS", f"{group}.ACS.apolar")

    pm.group(f"{group}.ACS.acceptor", f"{group}.ACS.acceptor.*")
    pm.group(f"{group}.ACS.donor", f"{group}.ACS.donor.*")
    pm.group(f"{group}.ACS.halogen", f"{group}.ACS.halogen.*")
    pm.group(f"{group}.ACS.aromatic", f"{group}.ACS.aromatic.*")
    pm.group(f"{group}.ACS.apolar", f"{group}.ACS.apolar.*")

    groups = [
        name
        for name in pm.get_names()
        if pm.get_type(name) == 'object:group'
            and name.startswith(f"{group}.")
            and pm.count_atoms(f"%{name}") == 0
    ]
    for grp in groups:
        if grp in pm.get_names():
            pm.delete(grp)
    
    ret = SimpleNamespace(
        clusters=clusters,
        eclusters=eclusters,
        hotspots=hotspots
    )
    return ret


@pm.extend
def count_molecules(sel):
    """
    Returns the number of distinct molecules in a given selection.
    """

    sel_copy = "__selcopy"
    pm.select(sel_copy, sel)
    num_objs = 0
    atoms_in_sel = pm.count_atoms(sel_copy)
    while atoms_in_sel > 0:
        num_objs += 1
        pm.select(sel_copy, "%s and not (bm. first %s)" % (sel_copy, sel_copy))
        atoms_in_sel = pm.count_atoms(sel_copy)
    return num_objs


@new_command
def get_fo(
    sel1: Selection,
    sel2: Selection,
    radius: float = 2,
    state1: int = 1,
    state2: int = 1,
    quiet: bool = True,
):
    """
    Compute the fractional overlap of sel1 respective to sel2.
        FO = Nc/Nt

    Nc is the number of atoms of sel1 in contact with sel2. Nt is the number of atoms
    of sel1. Hydrogen atoms are ignored.

    OPTIONS
        sel1    ligand object.
        sel2    hotspot object.
        state1  ligand state.
        state2  hotspot state.
        radius  the radius so sel1 and sel2 are in contact (default: 2).
    """
    xyz1 = get_coords(sel1, state=state1)
    xyz2 = get_coords(sel2, state=state2)
    if xyz1 is None or xyz2 is None:
        fo = 0
    else:
        dist = distance_matrix(xyz1, xyz2) <= radius
        nc = np.sum(np.any(dist, axis=1))
        nt = len(xyz1)
        fo = nc / nt
    if not quiet:
        print(f"FO: {fo:.2f}")
    return fo


@new_command
def get_dc(
    sel1: Selection,
    sel2: Selection,
    radius: float = 1.25,
    state1: int = 1,
    state2: int = 1,
    quiet: bool = True,
):
    """
    Compute the Density Correlation according to:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3264775/

    sel1 and sel2 are the selections representing the molecules or hotspots. The
    threshold distance can be changed with radius.

    OPTIONS
        sel1    first object
        sel2    second object
        state1  ligand state
        state2  hotspot state
        radius  the radius so two atoms are in contact (default: 1.25)

    EXAMPLES
        dc REF_LIG, ftmap1234.D_003_*_*
        dc ftmap1234.D.003, REF_LIG, radius=1.5

    """
    xyz1 = get_coords(sel1, state=state1)
    xyz2 = get_coords(sel2, state=state2)
    if xyz1 is None or xyz2 is None:
        dc = 0
    else:
        dc = (distance_matrix(xyz1, xyz2) < radius).sum()
    if not quiet:
        print(f"DC: {dc:.2f}")
    return dc


@new_command
def get_dce(
    sel1: Selection,
    sel2: Selection,
    radius: float = 1.25,
    state1: int = 1,
    state2: int = 1,
    quiet: bool = True,
):
    """
    Compute the Density Correlation Efficiency according to:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3264775/

    sel1 and sel2 are respectively the molecule and hotspot. The threshold
    distance can be changed with radius.

    OPTIONS
        sel1    ligand object
        sel2    hotspot object
        state1  ligand state
        state2  hotspot state
        radius  the radius so two atoms are in contact (default: 1.25)
        quiet   define verbosity

    EXAMPLE
        dce REF_LIG, ftmap1234.D_003_*_*
    """
    dce = get_dc(
        sel1,
        sel2,
        radius=radius,
        state1=state1,
        state2=state2
    ) / pm.count_atoms(sel1)
    if not quiet:
        print(f"DCE: {dce:.2f}")
    return dce


class LinkageMethod(StrEnum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"


@new_command
def get_ho(
    hs1: Selection,
    hs2: Selection,
    radius: float = 2.0,
    quiet: bool = True,
):
    """
    Compute the Hotspot Overlap (HO) metric. HO is defined as the number of
    atoms in hs1 in contact with hs2 plus the number of atoms in hs2 in
    contact with hs1 divided by the total number of atoms in both hotspots.

    OPTIONS
        hs1     an hotspot object
        hs2     another hotspot object
        radius  the distance to consider two atoms in contact (default: 2.5)
        quiet   define verbosity
    """
    atoms1 = get_coords(hs1)
    atoms2 = get_coords(hs2)
    if atoms1 is None or atoms2 is None:
        ho = 0
    else:
        dist = distance_matrix(atoms1, atoms2) <= radius
        num_contacts1 = np.sum(np.any(dist, axis=1))
        num_contacts2 = np.sum(np.any(dist, axis=0))
        ho = (num_contacts1 + num_contacts2) / (len(atoms1) + len(atoms2))
    if not quiet:
        print(f"HO: {ho:.2f}")
    return ho


@new_command
def plot_multivariate_hca(
    exprs: Selection,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: float = 0.0,
    only_medoids: bool = False,
    annotate: bool = False,
    plot: str = None,
    enable_heatmap: bool = False,
    rename_leafs: Optional[Dict[str, str]] = None,
    no_plot: bool = False,
):
    """
    Compute the similarity dendrogram of hotspots.
    OPTIONS
        exprs           space separated list of object expressions
        com_weight      center-of-mass (x, y, z) weight
        residue_radius  maximum distance for residue_similarity (default: 4)
        residue_weight  residue similarity weight (default: 1)
        residue_align   enable residue alignment (default: true)
        linkage_method  linkage method: single, complete or average
                        (default: single)
    EXAMPLES
        plot_similarity *.K15_D_* *.K15_DS_*, linkage_method=average
    """

    object_list = pm.get_object_list(exprs)
    assert object_list is not None and len(object_list) >= 2, "At least two hotspots are required for comparison."
    assert len(set(pm.get_property("Type", o) for o in object_list)) == 1, "Only hotspots of the same type are allowed in the HCA."

    hs_type = pm.get_property("Type", object_list[0])
    if hs_type == "HS":
        n_props = 5
    elif hs_type == "CS":
        n_props = 1
    elif hs_type == "ACS":
        n_props = 2
    labels = []

    p = np.zeros((len(object_list), n_props + 3))

    for ix, obj in enumerate(object_list):
        labels.append(obj)
        x, y, z = pm.centerofmass(obj)
        if hs_type == "HS":
            ST = pm.get_property("ST", obj)
            S0 = pm.get_property("S0", obj)
            CD0 = pm.get_property("CD0", obj)
            CD = pm.get_property("CD", obj)
            MD = pm.get_property("MD", obj)
            p[ix, :] = np.array([ST, S0, CD0, CD, MD, x, y, z])
        elif hs_type == "CS":
            ST = pm.get_property("ST", obj)
            p[ix, :] = np.array([ST, x, y, z])
        elif hs_type == "ACS":
            ST = pm.get_property("ST", obj)
            MD = pm.get_property("MD", obj)
            p[ix, :] = np.array([ST, MD, x, y, z])

    
    p = (p - p.mean(axis=0)) / (p.std(axis=0) + 1e-8)
    X = distance.pdist(p)
    dendro, medoids = plot_hca_base(X, labels, linkage_method, color_threshold, only_medoids, annotate, plot, enable_heatmap=enable_heatmap, rename_leafs=rename_leafs, no_plot=no_plot)
    return X, object_list, dendro, medoids


class OverlapFunction(StrEnum):
    FO = "fo"
    DC = "dc"
    DCE = "dce"


@new_command
def plot_overlap_matrix(
    sele_a: str,
    sele_b: Optional[str] = None,
    function: OverlapFunction = OverlapFunction.FO,
    radius: float = 2.0,
    annotate: bool = False

):
    objs_a = pm.get_object_list(sele_a)
    if sele_b.strip():
        objs_b = pm.get_object_list(sele_b)
    else:
        objs_b = objs_a
    
    match function:
        case OverlapFunction.FO:
            get_value = get_fo
        case OverlapFunction.DC:
            get_value = get_dc
        case OverlapFunction.DCE:
            get_value = get_dce

    ret = []
    X = []
    for i1, a in enumerate(objs_a):
        row = []
        for i2, b in enumerate(objs_b):
            value = get_value(a, b, radius=radius)
            row.append(value)
            ret.append([a, b, value])
        X.append(row)
    
    X = np.array(X)
    fig, ax = plt.subplots(constrained_layout=True)
    
    ax.set_yticks(range(len(objs_a)), objs_a)
    ax.set_xticks(range(len(objs_b)), objs_b)

    ax.tick_params(axis="x", rotation=90)
    if function == OverlapFunction.FO:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = None
        vmax = None
    image = ax.imshow(X, aspect="auto", vmin=vmin, vmax=vmax)
    if not annotate:
        fig.colorbar(image, ax=ax, shrink=0.8)

    if annotate:
        nan_mask = np.isnan(X)
        xmax = vmax or X[~nan_mask].max()
        xmin = vmin or X[~nan_mask].min()
        for i1, a in enumerate(objs_a):
            for i2, b in enumerate(objs_b):
                y = X[i1, i2]
                if np.isnan(y):
                    continue
                if (y - xmin)/(xmax - xmin) >= 0.5:
                    color = "black"
                else:
                    color = "white"
                if function == OverlapFunction.DC:
                    label = f"{y}"
                else:
                    label = f"{y:.2f}"
                ax.text(i2, i1, label, color=color, ha="center", va="center")
    return pd.DataFrame.from_records(ret, columns=['A', 'B', function.upper()])


def calc_medchem_bind_metrics(lig_sele: Selection, pki: float):
    mw = 0
    for at in pm.get_model(lig_sele).atom:
        mw += at.get_mass()
    ha = pm.count_atoms(f"({lig_sele}) and !elem H")
    bei = pki / mw
    le = pki / ha
    fq = le / (0.0715 + 7.5328/ha + 25.7079/ha**2 - 361.4722/ha**3)
    return {
        'sele': lig_sele,
        'ha': ha,
        'mw': mw,
        'pki': pki,
        'bei': bei,
        'le': le,
        'fq': fq
    }


class BindMetric(StrEnum):
    PKI = "pki"
    LE = "le"
    BEI = "bei"
    FQ = "fq"


def plot_ligand_fit(
    hs_sele: Selection,
    ligs_sele: Selection,
    function: OverlapFunction,
    radius: float,
    annotate: bool,
    lig_metric: BindMetric,
    bind_df
):
    if len(pm.get_object_list(hs_sele)) != 1:
        raise ValueError("Only one hotspot can be analyzed at time.")
    overlap_df = plot_overlap_matrix(
        sele_a=hs_sele,
        sele_b=ligs_sele,
        function=function,
        radius=radius,
        annotate=annotate
    ).rename(columns={'B': 'Ligand'})
    plt.close()

    # identify the fragment
    ix_frag = np.argmin(bind_df['HA'])
    
    # merge dataframes
    bind_df.rename(columns={'sele': 'Ligand'})
    df = overlap_df.join(bind_df, on='Ligand', how='left')
    function_col = function.upper()

    # do the actual plot
    lig_metric = lig_metric.upper()
    fig, ax = plt.subplots(constrained_layout=True)
    x = df[function_col] / df[function_col].iloc[ix_frag]
    y = df[lig_metric] / df[lig_metric].iloc[ix_frag]
    ax.scatter(x, y)
    rows = zip(x, y, df['Ligand'], df['Label'])
    for x, y, obj, label in rows:
        s = label.strip() or obj
        ax.text(x, y, s)
    ax.set_xlabel(f"{function_col} / {function_col}_ref")
    ax.set_ylabel(f"{lig_metric} / {lig_metric}_ref")

#
# GRAPHICAL USER INTERFACE
#

from pymol import Qt

QWidget = Qt.QtWidgets.QWidget
QScrollArea = Qt.QtWidgets.QScrollArea
QFileDialog = Qt.QtWidgets.QFileDialog
QFormLayout = Qt.QtWidgets.QFormLayout
QPushButton = Qt.QtWidgets.QPushButton
QSpinBox = Qt.QtWidgets.QSpinBox
QDoubleSpinBox = Qt.QtWidgets.QDoubleSpinBox
QLineEdit = Qt.QtWidgets.QLineEdit
QCheckBox = Qt.QtWidgets.QCheckBox
QVBoxLayout = Qt.QtWidgets.QVBoxLayout
QHBoxLayout = Qt.QtWidgets.QHBoxLayout
QDialog = Qt.QtWidgets.QDialog
QComboBox = Qt.QtWidgets.QComboBox
QTabWidget = Qt.QtWidgets.QTabWidget
QLabel = Qt.QtWidgets.QLabel
QTableWidget = Qt.QtWidgets.QTableWidget
QTableWidgetItem = Qt.QtWidgets.QTableWidgetItem
QGroupBox = Qt.QtWidgets.QGroupBox
QHeaderView = Qt.QtWidgets.QHeaderView
QStyledItemDelegate = Qt.QtWidgets.QStyledItemDelegate
QShortcut = Qt.QtWidgets.QShortcut

QtCore = Qt.QtCore
QLocale = Qt.QtCore.QLocale
QIcon = Qt.QtGui.QIcon
QDoubleValidator = Qt.QtGui.QDoubleValidator
QKeySequence = Qt.QtGui.QKeySequence
QValidator = Qt.QtGui.QValidator
QApplication = Qt.QtWidgets.QApplication

class LoadWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Group", "Filename"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        addRemoveLayout = QHBoxLayout()
        layout.addLayout(addRemoveLayout)

        pickFileButton = QPushButton("Add")
        pickFileButton.clicked.connect(self.pickFile)
        addRemoveLayout.addWidget(pickFileButton)

        removeButton = QPushButton("Remove")
        removeButton.clicked.connect(self.removeRow)
        addRemoveLayout.addWidget(removeButton)

        loadButton = QPushButton("Load")
        loadButton.clicked.connect(self.load)
        addRemoveLayout.addWidget(loadButton)

        groupBox = QGroupBox("Options")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)
            
        self.maxCollisions = QDoubleSpinBox()
        self.maxCollisions.setRange(0.0, 1.0)
        self.maxCollisions.setSingleStep(0.05)
        self.maxCollisions.setValue(0.10)
        boxLayout.addRow("Max collisions:", self.maxCollisions)
        
    def pickFile(self):
        fileDIalog = QFileDialog()
        fileDIalog.setFileMode(QFileDialog.ExistingFiles)
        fileDIalog.setNameFilter("FTMap PDB (*.pdb)")
        fileDIalog.setViewMode(QFileDialog.Detail)

        if fileDIalog.exec_():
            for filename in fileDIalog.selectedFiles():
                basename = os.path.splitext(os.path.basename(filename))
                group = basename[0]
                self.appendRow(filename, group)

    def appendRow(self, filename, group):
        groupItem = QTableWidgetItem(group)
        filenameItem = QTableWidgetItem(filename)

        filenameItem.setFlags(filenameItem.flags() & ~QtCore.Qt.ItemIsEditable)

        self.table.insertRow(self.table.rowCount())
        self.table.setItem(self.table.rowCount() - 1, 0, groupItem)
        self.table.setItem(self.table.rowCount() - 1, 1, filenameItem)

    def removeRow(self):
        self.table.removeRow(self.table.currentRow())

    def clearInputs(self):
        self.table.setRowCount(0)

    def load(self):
        cd_to_anchor = False
        combinatory_search = True
        allow_nested = True
        max_collisions = self.maxCollisions.value()
        try:
            filenames = []
            groups = []
            for row in range(self.table.rowCount()):
                filename = self.table.item(row, 1).text()
                group = self.table.item(row, 0).text()
                filenames.append(filename)
                groups.append(group)
        finally:
            self.clearInputs()
        
        load_ftmap(
            filenames,
            groups=groups,
            cd_to_anchor=cd_to_anchor,
            combinatory_search=combinatory_search,
            allow_nested=allow_nested,
            max_collisions=max_collisions,
        )


class SortableItem(QTableWidgetItem):
    def __init__(self, obj):
        super().__init__()
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEditable)
        try:
            self.setData(QtCore.Qt.ItemDataRole.EditRole, float(obj))
        except:
            self.setData(QtCore.Qt.ItemDataRole.EditRole, str(obj))
        try:
            self.setData(QtCore.Qt.ItemDataRole.DisplayRole, f"{obj:.2f}")
        except (ValueError, TypeError):
            self.setData(QtCore.Qt.ItemDataRole.DisplayRole, str(obj))


class OptionalPositiveDoubleDelegate(QStyledItemDelegate):
    class Validator(QDoubleValidator):
        def validate(self, string, pos):
            # Se a string estiver vazia, permitimos (retornamos Acceptable)
            if not string:
                return QValidator.Acceptable, string, pos
            
            # Caso contrário, usamos a validação padrão de números
            try:
                float(string) > 0
                return QValidator.Acceptable, string, pos
            except ValueError:
                return super().validate(string, pos)
            
    
    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        
        validator = self.Validator(editor)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        
        editor.setValidator(validator)
        return editor



    

class TableWidget(QWidget):

    class TableWidgetImpl(QTableWidget):
        def __init__(self, props):
            super().__init__()
            self.setSelectionBehavior(QTableWidget.SelectRows)
            self.setSelectionMode(QTableWidget.SingleSelection)
            self.setColumnCount(len(props) + 1)
            self.setHorizontalHeaderLabels(["Object"] + props)
            header = self.horizontalHeader()
            for idx in range(len(props) + 1):
                header.setSectionResizeMode(
                    idx, QHeaderView.ResizeMode.ResizeToContents
                )

            @self.itemSelectionChanged.connect
            def itemSelectionChanged():
                for item in self.selectedItems():
                    obj = self.item(item.row(), 0).text()
                    pm.select(obj)
                    pm.enable("sele")
                    break

    def __init__(self):
        super().__init__()
        self.selected_objs = set()
        self.current_tab = "HS"

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.filter_line = QLineEdit("*")
        self.filter_line.setPlaceholderText("PyMOL Selection Algebra")
        layout.addWidget(self.filter_line)

        @self.filter_line.textEdited.connect
        def textEdited(expr):
            self.current_tab = [k[1] for k in self.hotspotsMap.keys()][tab.currentIndex()]
            self.updateCurrentList()
    

        tab = QTabWidget()
        layout.addWidget(tab)

        self.hotspotsMap = {
            ("Hotspots", "HS"): [
                "Class",
                "ST",
                "S0",
                "CD",
                "MD",
                "Length",
            ],
            ("CS", "CS"): ["ST"],
        }
        self.tables = {}
        for (title, key), props in self.hotspotsMap.items():
            table = self.TableWidgetImpl(props)
            self.tables[title] = table
            tab.addTab(table, title)

        @tab.currentChanged.connect
        def currentChanged(tab_index):
            self.changeItems(tab_index)

        exportButton = QPushButton(QIcon("save"), "Export Tables")
        exportButton.clicked.connect(self.export)
        layout.addWidget(exportButton)

    def changeItems(self, tab_index):
        self.current_tab = [k[1] for k in self.hotspotsMap.keys()][tab_index]
        self.updateCurrentList()
    
    def showEvent(self, event):
        self.filter_line.textEdited.emit(self.filter_line.text())
        self.refresh()
        super().showEvent(event)

    def refresh(self):
        for (title, key), props in self.hotspotsMap.items():
            self.tables[title].setSortingEnabled(False)

            # remove old rows
            while self.tables[title].rowCount() > 0:
                self.tables[title].removeRow(0)

            # append new rows
            for obj in pm.get_names("objects"):
                if pm.get_type(obj) in ['object:group', 'selection']:
                    continue
                if not pm.get_property_list(obj):
                    continue
                obj_type = pm.get_property("Type", obj)
                if obj_type == key:
                    if obj in self.selected_objs:
                        self.appendRow(title, key, obj)

            self.tables[title].setSortingEnabled(True)

    def appendRow(self, title, key, obj):
        self.tables[title].insertRow(self.tables[title].rowCount())
        line = self.tables[title].rowCount() - 1

        self.tables[title].setItem(line, 0, SortableItem(obj))

        for idx, prop in enumerate(self.hotspotsMap[(title, key)]):
            prop_value = pm.get_property(prop, obj)
            self.tables[title].setItem(line, idx + 1, SortableItem(prop_value))

    def updateCurrentList(self):
        sele = self.filter_line.text()
        if sele.strip() == '':
            sele = '*'
        self.selected_objs = pm.get_object_list(sele)
        if self.selected_objs is None:
            self.selected_objs = set()
        for obj in self.selected_objs.copy():
            if pm.get_property("Type", obj) != self.current_tab:
                self.selected_objs.remove(obj)
        self.refresh()
    
    def export(self):
        fileDialog = QFileDialog()
        fileDialog.setNameFilter("Excel file (*.xlsx)")
        fileDialog.setViewMode(QFileDialog.Detail)
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        fileDialog.setDefaultSuffix(".xlsx")

        if fileDialog.exec_():
            filename = fileDialog.selectedFiles()[0]
            ext = os.path.splitext(filename)[1]
            with pd.ExcelWriter(filename) as xlsx_writer:
                prev_filter_text = self.filter_line.text()
                prev_tab = self.current_tab
                self.filter_line.setText("*")
                try:
                    for (title, key), props in self.hotspotsMap.items():
                        data = {"Object": [], **{p: [] for p in props}}
                        for header in data:
                            self.current_tab = key
                            self.updateCurrentList()

                            column = list(data.keys()).index(header)
                            for line in range(self.tables[title].rowCount()):
                                item = self.tables[title].item(line, column)
                                data[header].append(self.parse_item(item))
                        df = pd.DataFrame(data)
                        df.to_excel(xlsx_writer, sheet_name=title, index=False)
                finally:
                    self.filter_line.setText(prev_filter_text)
                    self.current_tab = prev_tab
    @staticmethod
    def parse_item(item):
        try:
            item = int(item.text())
        except ValueError:
            try:
                item = float(item.text())
            except ValueError:
                item = item.text()
        return item


class HcaWidget(QWidget):

    def __init__(self):
        super().__init__()

        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        tab = QTabWidget()
        mainLayout.addWidget(tab)

        groupBox = QWidget()
        mainLayout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)
        tab.addTab(groupBox, "General")

        self.hotspotSeleLine = QLineEdit()
        self.hotspotSeleLine.setPlaceholderText("Objects or selection")
        boxLayout.addRow("Hotspots:", self.hotspotSeleLine)

        self.linkageMethodCombo = QComboBox()
        self.linkageMethodCombo.addItems([e.value for e in LinkageMethod])
        boxLayout.addRow("Linkage:", self.linkageMethodCombo)

        self.colorThresholdSpin = QDoubleSpinBox()
        self.colorThresholdSpin.setMinimum(0)
        self.colorThresholdSpin.setMaximum(100)
        self.colorThresholdSpin.setValue(0)
        self.colorThresholdSpin.setSingleStep(0.1)
        self.colorThresholdSpin.setDecimals(2)
        boxLayout.addRow("Color threshold:", self.colorThresholdSpin)

        self.onlyMedoidsCheck = QCheckBox()
        self.onlyMedoidsCheck.setChecked(False)
        boxLayout.addRow("Show only medoids:", self.onlyMedoidsCheck)

        self.enableHeatmapCheck = QCheckBox()
        boxLayout.addRow("Heatmap:", self.enableHeatmapCheck)

        self.annotateCheck = QCheckBox()
        boxLayout.addRow("Annotate heatmap:", self.annotateCheck)

        self.table = QTableWidget()
        tab.addTab(self.table, "Leaf names")
        # self.table.setSelectionBehavior(QTableWidget.SelectRows)
        # self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Name", "Rename"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(""))
        self.table.setItem(0, 1, QTableWidgetItem(""))
        self.shortcut = QShortcut(QKeySequence("Ctrl+V"), self.table)
        self.shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.shortcut.activated.connect(self.paste_data)
        @self.table.itemChanged.connect
        def itemChanged(item):
            self.table.blockSignals(True)
            for row in range(self.table.rowCount()-1, -1, -1):
                col0_text = self.table.item(row, 0)
                col0_text = col0_text.text().strip()
                
                col1_text = self.table.item(row, 1)
                col1_text = col1_text.text().strip()
                
                if (col0_text == "" and col1_text == ""):
                    self.table.removeRow(row)
            
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(""))
            self.table.setItem(row, 1, QTableWidgetItem(""))
            
            self.table.blockSignals(False)

        layout = QHBoxLayout()
        container = QWidget(self)
        container.setLayout(layout)
        mainLayout.addWidget(container)
        
        groupBox = QGroupBox("Multivariate analysis")
        
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.multivariateFunctionCombo = QComboBox()
        self.multivariateFunctionCombo.addItems(["euclidean"])

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_multivariate_hca)
        boxLayout.addWidget(plotButton)

    def paste_data(self):
        if not self.table.isVisible() or not self.table.isEnabled():
            return
        
        clipboard = QApplication.instance().clipboard()
        mime_data = clipboard.mimeData()
        if not mime_data.hasText():
            return
        text = mime_data.text()
        rows = text.strip().split('\n')

        self.table.blockSignals(True)
        while self.table.rowCount() > 0:
            self.table.removeRow(0)
        
        for row_ix, row_text in enumerate(rows):
            columns = row_text.split('\t')
            self.table.insertRow(self.table.rowCount())

            for col_ix, col_text in enumerate(columns):
                if col_ix < self.table.columnCount():
                    value = col_text.strip().replace(',', '.')
                    item = QTableWidgetItem(value)
                    self.table.setItem(row_ix, col_ix, item)
        self.table.insertRow(self.table.rowCount())
        self.table.setItem(row_ix+1, 0, QTableWidgetItem(""))
        self.table.setItem(row_ix+1, 1, QTableWidgetItem(""))
        self.table.blockSignals(False)
    
    def getLeafLabels(self):
        rows = self.table.rowCount()
        data = {}
        for r in range(rows):
            obj = self.table.item(r, 0).text().strip()
            lbl = self.table.item(r, 1).text().strip()
            if obj == "" or lbl == "":
                continue
            data[obj] = lbl
        return data
    
    def plot_multivariate_hca(self):
        sele = self.hotspotSeleLine.text()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        only_medoids = self.onlyMedoidsCheck.isChecked()
        annotate = self.annotateCheck.isChecked()
        enable_heatmap = self.enableHeatmapCheck.isChecked()

        plot_multivariate_hca(
            exprs=sele,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            only_medoids=only_medoids,
            annotate=annotate,
            enable_heatmap=enable_heatmap,
            rename_leafs=self.getLeafLabels(),
        )


class LigandTableWidget(QTableWidget):
    COLUMNS = ["Ligand", "PKI", "LE", "BEI", "FQ", "HA", "MW", "Label"]
    def __init__(self):
        super().__init__()
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        header = self.horizontalHeader()
        for idx in range(len(self.COLUMNS)):
            header.setSectionResizeMode(
                idx, QHeaderView.ResizeMode.ResizeToContents
            )
        self.setItemDelegateForColumn(1, OptionalPositiveDoubleDelegate())

        @self.itemSelectionChanged.connect
        def itemSelectionChanged():
            for item in self.selectedItems():
                obj = self.item(item.row(), 0).text()
                pm.select(obj)
                pm.enable("sele")
                break
    

    def refresh(self, objects):
        self.setSortingEnabled(False)
        self.blockSignals(True)
        
        while self.rowCount() > 0:
            self.removeRow(0)
        for obj in objects:
            self.insertRow(self.rowCount())
            row = self.rowCount() - 1

            item = QTableWidgetItem(obj)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.setItem(row, 0, item)

            self.setItem(row, 1, QTableWidgetItem(""))
            for col in range(2, 7):
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(row, col, item)
            
            self.setItem(row, 7, QTableWidgetItem(""))
                
        self.setSortingEnabled(True)
        self.blockSignals(False)
    
    def getDataFrame(self):
        rows = self.rowCount()
        cols = len(self.COLUMNS)
        headers = [
            self.horizontalHeaderItem(c).text()
            for c in range(cols)
        ]
        data = []
        for r in range(rows):
            obj = self.item(r, 0).text()
            pki = self.item(r, 1).text()
            ha = self.item(r, 5).text()
            mw = self.item(r, 6).text()
            label = self.item(r, 7).text()
            if not pki:
                data.append([
                    obj,
                    "",
                    "",
                    "",
                    "",
                    int(ha),
                    float(mw.replace(",", ".")),
                    label,
                ])
            else:
                le = self.item(r, 2).text()
                bei = self.item(r, 3).text()
                fq = self.item(r, 4).text()
                
                data.append([
                    obj,
                    float(pki.replace(",", ".")),
                    float(le.replace(",", ".")),
                    float(bei.replace(",", ".")),
                    float(fq.replace(",", ".")),
                    int(ha),
                    float(mw.replace(",", ".")),
                    label,
                ])
        return pd.DataFrame(data, columns=headers)


class LigandFitWidget(QWidget):

    def __init__(self):
        super().__init__()
        
        layout = QFormLayout()
        self.setLayout(layout)
        
        self.hotspotsSeleLine = QLineEdit()
        self.hotspotsSeleLine.setPlaceholderText("Single hotspot object or selection...")
        self.hotspotsSeleLine.textChanged.connect(self.validateUpdateWidget)
        layout.addRow("Hotspots:", self.hotspotsSeleLine)

        self.ligandsSeleLine = QLineEdit()
        self.ligandsSeleLine.setPlaceholderText("Ligand objects or selections...")
        self.ligandsSeleLine.textChanged.connect(self.validateUpdateWidget)
        layout.addRow("Ligands:", self.ligandsSeleLine)
        
        self.ligMetricCombo = QComboBox()
        self.ligMetricCombo.addItems([e.value for e in BindMetric if e != BindMetric.PKI])
        layout.addRow("Binding metric:", self.ligMetricCombo)
        
        self.functionCombo = QComboBox()
        self.functionCombo.addItems([e.value for e in OverlapFunction])
        layout.addRow("Overlap function:", self.functionCombo)

        self.radiusSpin = QDoubleSpinBox()
        self.radiusSpin.setValue(2)
        self.radiusSpin.setSingleStep(0.5)
        self.radiusSpin.setDecimals(2)
        self.radiusSpin.setMinimum(1)
        self.radiusSpin.setMaximum(10)
        layout.addRow("Radius:", self.radiusSpin)

        self.annotateCheck = QCheckBox()
        self.annotateCheck.setChecked(True)
        layout.addRow("Annotate:", self.annotateCheck)

        self.table = LigandTableWidget()
        self.table.setEnabled(False)
        layout.addRow(self.table)
        @self.table.itemChanged.connect
        def itemChanged(item):
            self.table.setSortingEnabled(False)
            row = item.row()

            if item.column() == 1:
                if item.text() == "":
                    self.table.item(row, 2).setText(f"")
                    self.table.item(row, 3).setText(f"")
                    self.table.item(row, 4).setText(f"")
                    # self.table.item(row, 5).setText(f"")
                    # self.table.item(row, 6).setText(f"")
                else:
                    lig_obj = self.table.item(row, 0).text()
                    pki = float(self.table.item(row, 1).text())
                    bind = calc_medchem_bind_metrics(lig_obj, pki)
                    le = bind['le']
                    bei = bind['bei']
                    fq = bind['fq']
                    ha = bind['ha']
                    mw = bind['mw']
                
                    self.table.item(row, 2).setText(f"{le:.3f}")
                    self.table.item(row, 3).setText(f"{bei:.3f}")
                    self.table.item(row, 4).setText(f"{fq:.3f}")
                    self.table.item(row, 5).setText(f"{ha}")
                    self.table.item(row, 6).setText(f"{mw:.2f}")
            self.table.setSortingEnabled(True)

        self.container = QWidget()
        hLayout = QHBoxLayout(self.container)
        layout.addRow(self.container)

        self.plotButton = QPushButton("Plot")
        self.plotButton.clicked.connect(self.plot)
        hLayout.addWidget(self.plotButton)

        self.exportButton = QPushButton("Export")
        self.exportButton.clicked.connect(self.export)
        hLayout.addWidget(self.exportButton)

    def validateUpdateWidget(self):
        hs_sele = self.hotspotsSeleLine.text().strip()
        hs_objs = pm.get_object_list(hs_sele)
        if hs_objs is None:
            hs_objs = []
        if len(hs_objs) == 0:
            self.container.setEnabled(False)
        else:
            self.container.setEnabled(True)

        ligs_sele = self.ligandsSeleLine.text().strip()
        ligs_objs = pm.get_object_list(ligs_sele)
        if ligs_objs is None:
            ligs_objs = []
        self.table.refresh(ligs_objs)
        if len(ligs_objs) > 0:
            self.table.setEnabled(True)
            self.exportButton.setEnabled(True)
            if len(hs_objs) == 1:
                self.plotButton.setEnabled(True)
        else:
            self.table.setEnabled(False)
            self.plotButton.setEnabled(False)
            self.exportButton.setEnabled(False)
        if len(hs_objs) != 1:
            self.plotButton.setEnabled(False)

    def plot(self):
        hs_sele = self.hotspotsSeleLine.text().strip()
        ligs_sele = self.ligandsSeleLine.text().strip()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        annotate = self.annotateCheck.isChecked()
        lig_metric = self.ligMetricCombo.currentText()

        bind_df = self.table.getDataFrame()
        bind_df = bind_df.set_index('Ligand')

        plot_ligand_fit(
            hs_sele=hs_sele,
            ligs_sele=ligs_sele,
            function=function,
            radius=radius,
            annotate=annotate,
            lig_metric=lig_metric,
            bind_df=bind_df
        )
        plt.show()
    
    def export(self):
        ligand_df = self.table.getDataFrame()

        fileDialog = QFileDialog()
        fileDialog.setNameFilter("Excel file (*.xlsx)")
        fileDialog.setViewMode(QFileDialog.Detail)
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        fileDialog.setDefaultSuffix(".xlsx")

        if fileDialog.exec_():
            filename = fileDialog.selectedFiles()[0]
            # ext = os.path.splitext(filename)[1]
            with pd.ExcelWriter(filename) as xlsx_writer:
                ligand_df.to_excel(xlsx_writer, sheet_name='Ligand', index=False)


class OverlapWidget(QWidget):

    def __init__(self):
        super().__init__()
        
        layout = QFormLayout()
        self.setLayout(layout)

        self.aSeleLine = QLineEdit()
        self.aSeleLine.setPlaceholderText("Objects or selections...")
        layout.addRow("Selection A:", self.aSeleLine)

        self.bSeleLine = QLineEdit()
        self.bSeleLine.setPlaceholderText("Objects or selections...")
        layout.addRow("Selection B:", self.bSeleLine)
        
        self.functionCombo = QComboBox()
        self.functionCombo.addItems([e.value for e in OverlapFunction])
        layout.addRow("Overlap function:", self.functionCombo)

        self.radiusSpin = QDoubleSpinBox()
        self.radiusSpin.setValue(2)
        self.radiusSpin.setSingleStep(0.5)
        self.radiusSpin.setDecimals(2)
        self.radiusSpin.setMinimum(1)
        self.radiusSpin.setMaximum(10)
        layout.addRow("Radius:", self.radiusSpin)

        self.annotateCheck = QCheckBox()
        self.annotateCheck.setChecked(True)
        layout.addRow("Annotate:", self.annotateCheck)

        self.container1 = QWidget()
        hLayout1 = QHBoxLayout(self.container1)
        layout.addRow(self.container1)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_overlap)
        hLayout1.addWidget(plotButton)

        exportButton = QPushButton("Export")
        exportButton.clicked.connect(self.export_overlap)
        hLayout1.addWidget(exportButton)

    def plot_overlap(self):
        sele_a = self.aSeleLine.text().strip()
        sele_b = self.bSeleLine.text().strip()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        annotate = self.annotateCheck.isChecked()
        
        plot_overlap_matrix(
            sele_a=sele_a,
            sele_b=sele_b,
            function=function,
            radius=radius,
            annotate=annotate
        )
        plt.show()
    
    def export_overlap(self):
        sele_a = self.aSeleLine.text().strip()
        sele_b = self.bSeleLine.text().strip()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        annotate = self.annotateCheck.isChecked()
        
        table_df = plot_overlap_matrix(
            sele_a=sele_a,
            sele_b=sele_b,
            function=function,
            radius=radius,
            annotate=annotate
        )
        plt.close()

        fileDialog = QFileDialog()
        fileDialog.setNameFilter("Excel file (*.xlsx)")
        fileDialog.setViewMode(QFileDialog.Detail)
        fileDialog.setAcceptMode(QFileDialog.AcceptSave)
        fileDialog.setDefaultSuffix(".xlsx")

        if fileDialog.exec_():
            filename = fileDialog.selectedFiles()[0]
            # ext = os.path.splitext(filename)[1]
            with pd.ExcelWriter(filename) as xlsx_writer:
                table_df.to_excel(xlsx_writer, sheet_name='Overlap', index=False)



class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(650, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("(XDrugPy) Hotspots")

        tab = QTabWidget()
        tab.addTab(LoadWidget(), "Load")
        tab.addTab(TableWidget(), "Properties")
        tab.addTab(HcaWidget(), "Hotspot Similarity")
        tab.addTab(OverlapWidget(), "Overlap Matrix")
        tab.addTab(LigandFitWidget(), "Ligand Fit")

        layout.addWidget(tab)



dialog = None

def run_plugin_gui():
    global dialog
    if dialog is None:
        
        locale = QLocale(QLocale.English, QLocale.UnitedStates)
        QLocale.setDefault(locale)

        dialog = MainDialog()
        dialog.setLocale(locale)
    dialog.show()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("(XDrugPy) Hotspots", run_plugin_gui)
