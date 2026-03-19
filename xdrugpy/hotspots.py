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
from scipy.spatial import distance_matrix, distance, cKDTree, ConvexHull

from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from strenum import StrEnum
import networkx as nx
import pyKVFinder

from . import XDRUGPY_EXPERIMENTAL_VERSION
from .utils import (
    new_command,
    Selection,
    plot_hca_base,
    clustal_omega,
    run,
    Residue
)

from pymol import cmd as pm
from pymol.exporting import _resn_to_aa as RESN_TO_AA


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


def get_clusters():
    clusters = []
    eclusters = []
    for obj in pm.get_object_list():
        if obj.startswith(f"crosscluster."):
            _, _, s, _ = obj.split(".", maxsplit=4)
            pm.remove(f"%{obj} & elem H")
            coords = pm.get_coords(obj)
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
            coords = pm.get_coords(obj)
            clusters.append(
                Cluster(
                    selection=obj,
                    coords=coords,
                    ST=int(s),
                )
            )
        elif obj.startswith("clust."):
            _, idx, s, probe_type = obj.split(".", maxsplit=4)
            coords = pm.get_coords(obj)
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

        
        hs_sele = f"{group}.CS.* near_to 4 of ({p_sele})"
        sele_cnt = 0
        while True:
            new_sele_cnt = pm.count_atoms(hs_sele)
            if new_sele_cnt != sele_cnt:
                sele_cnt = new_sele_cnt
            else:
                break
            hs_sele = f"{group}.CS.* within 4 of ({hs_sele})"
        
        hs_objs = pm.get_object_list(hs_sele)
        if not hs_objs:
            continue
                
        pocket_clusters = [c for c in clusters if c.selection in hs_objs]
        ix_ignore = next((i for i, c in enumerate(pocket_clusters) if c.ST<5), len(pocket_clusters))
        pocket_clusters = pocket_clusters[:ix_ignore]

        if not pocket_clusters:
            continue

        # p_xyz = pm.get_coords(p_sele)
        # hs_xyz = pm.get_coords(hs_sele)
        # if len(hs_xyz) < 4:
        #     continue

        # hull = ConvexHull(hs_xyz)
        # surf_ixs = hull.vertices
        # surf_xyz = hs_xyz[surf_ixs]
        
        # d = distance_matrix(surf_xyz, p_xyz) < 3
        # nc = np.sum(np.any(d, axis=1))
        # if nc < 0.2 * len(surf_xyz):
        #     continue
        
        hs_sele = ' | '.join([c.selection for c in pocket_clusters])
        if hs_sele in pockets:
            assert all(c1 == c2 for c1, c2 in zip(pocket_clusters, pockets[hs_sele]))
            assert len(pocket_clusters) == len(pockets[hs_sele])
        pockets[hs_sele] = pocket_clusters
    return pockets


def find_pykvf_pockets(protein):
    with tempfile.TemporaryDirectory() as tempdir:
        protein_pdb = f"{tempdir}/protein.pdb"
        pm.save(protein_pdb, selection=protein)
        atomic = pyKVFinder.read_pdb(protein_pdb)
        
    vertices = pyKVFinder.get_vertices(atomic)
    _, cavities = pyKVFinder.detect(atomic, vertices, volume_cutoff=750)
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
    MD: float
    length: int
    isComplex: bool
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
    def from_cluster_selections(selections: List[str], max_collisions: float=0.25) -> Hotspot:
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
        return Hotspot.from_clusters(group, clusters, max_collisions=max_collisions)
        
    @staticmethod
    def from_clusters(group: str, clusters: List[Cluster], max_collisions: float=0.25) -> Hotspot:
        coms = [pm.centerofmass(c.selection) for c in clusters]
        cd = [distance.euclidean(coms[0], com) for com in coms]

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
            SZ=clusters[-1].ST if len(clusters) >= 3 else 0,
            CD=np.max(cd),
            MD=max_dist,
            length=len(clusters),
            isComplex=len(clusters) >= 4 and clusters[1].ST >= 16,
            nComponents=-1,
        )
        
        s0 = hs.S0
        sz = hs.SZ
        cd = hs.CD
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
        allow_nested: bool,
        max_collisions: float=0.25
    ) -> List[Hotspot]:
        
        # filter out weak clusters
        ix_ignore = next((i for i, c in enumerate(clusters) if c.ST<5), -1)
        clusters = clusters[:ix_ignore]

        # identify hotspots from pockets and consensus sites
        spots = []
        pockets = find_occupied_pockets(group, pocket_residues, clusters)
        for hs_sele, pocket_clusters in pockets.items():
            hs = Hotspot.from_clusters(group, pocket_clusters, max_collisions=max_collisions)
            if hs.klass:
                hs.selection = hs_sele
                spots.append(hs)
        
        # identify hotspots from combinations of consensus sites
        for r in range(1, 4):
            for comb in combinations(clusters, r):
                comb = list(comb)
                hs = Hotspot.from_clusters(group, comb, max_collisions=max_collisions)
                if hs.klass:
                    spots.append(hs)
        
        # remove hotspot objects when they totally fit inside another (and are of the same class)
        nested = set()
        if not allow_nested:
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
        spots = sorted(spots, key=lambda hs: (-hs.S0, -hs.S1, -hs.SZ, -hs.ST))
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

    def make_graph(group: str, clusters: List[Cluster], radius: float=0.5, samples: int=50, max_collisions: float=0.25) -> int:
        g = nx.Graph()
        # Adiciona todos os nomes de clusters como NÓS (essencial para contar isolados)
        g.add_nodes_from([c.selection for c in clusters])

        for ix1, c1 in enumerate(clusters):
            for ix2, c2 in enumerate(clusters):
                if ix1 >= ix2:
                    continue
                collision = Hotspot.has_collision(group, c1.selection, c2.selection, radius, samples)
                # Lógica de Conectividade:
                # Um átomo em A está 'conectado' a B se ele tem pelo menos UM caminho livre para B
                atoms1_collided = np.any(collision, axis=1) # Vetor (N,)
                atoms2_collided = np.any(collision, axis=0) # Vetor (M,)
                
                # Se mais de 75% dos átomos de ambos os clusters conseguem se 'ver', estão no mesmo bolso
                if np.mean(atoms1_collided) < max_collisions and np.mean(atoms2_collided) < max_collisions:
                    g.add_edge(c1.selection, c2.selection)
        return nx.number_connected_components(g)

    @staticmethod
    @lru_cache
    def has_collision(group: str, clu_sele1: str, clu_sele2: str, radius: float, samples: int):

        list_a = pm.get_coords(clu_sele1)
        list_b = pm.get_coords(clu_sele2)
        
        prot_xyz = pm.get_coords(f'{group}.protein')
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
def show_hs(selections: List[str]) -> Hotspot:
    hs = Hotspot.from_cluster_selections(selections, max_collisions=0.20)
    hs.show()
    print(hs)
    return hs


class FTMapResults:
    pass

@new_command
def load_ftmap(
    filenames: List[Path] | Path,
    groups: List[str] | str = "",
    allow_nested: bool = False,
    max_collisions: float = 0.20,
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
                rets.append(_load_ftmap(fnames, groups, allow_nested=allow_nested, max_collisions=max_collisions))
            except:
                rets.append(_load_ftmap(fnames, groups, allow_nested=allow_nested, max_collisions=max_collisions))
        if single:
            return rets[0]
        else:
            return rets
    finally:
        pm.set('defer_updates', 0)


def _load_ftmap(
    filename: Path,
    group: str = "",
    allow_nested: bool = False,
    max_collisions: float = 0.20,
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
    pm.load(str(filename), quiet=1)

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
    hotspots = Hotspot.find_hotspots(group, pocket_residues, clusters, allow_nested, max_collisions=max_collisions)

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
    xyz1 = pm.get_coords(sel1)
    xyz2 = pm.get_coords(sel2)
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
    xyz1 = pm.get_coords(sel1)
    xyz2 = pm.get_coords(sel2)
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
    dce = get_dc(sel1, sel2, radius=radius) / pm.count_atoms(sel1)
    if not quiet:
        print(f"DCE: {dce:.2f}")
    return dce


class LinkageMethod(StrEnum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"


@new_command
def fpt_sim(
    multi_seles: Selection,
    site: Selection = "*",
    site_radius: float = 5.0,
    seq_align_omega: bool = False,
    omega_conservation: str = "*:.",
    contact_radius: float = 4.0,
    nbins: int = 5,
    sharex: bool = True,
    linkage_method: LinkageMethod = LinkageMethod.WARD,
    color_threshold: float = 0.0,
    only_medoids: bool = False,
    annotate: bool = True,
    plot_fingerprints: str = "",
    share_ylim: bool = True,
    plot_hca: str = "",
    quiet: bool = True,
):
    """
    Compute the similarity between the residue contact fingerprint of two
    hotspots.

    OPTIONS:
        hotspots          hotspot selection
        site              selection to focus based on first protein
        radius            radius to compute the contacts (default: 4)
        plot_fingerprints plot the fingerprints (default: True)
        nbins             number of residue labels (default: 5)
        plot_dendrogram   plot the dendrogram (default: False)
        linkage_method    linkage method (default: single)
        quiet             define verbosity

    EXAMPLES
        fs_sim 8DSU.K15_D_01* 6XHM.K15_D_01*
        fs_sim 8DSU.CS_* 6XHM.CS_*, site=resi 8-101, nbins=10
    """
    seles = []
    groups = []

    for sele in multi_seles.split("/"):
        sele = sele.strip()
        seles.append(sele.strip())
        obj = pm.get_object_list(sele)
        if obj is not None and len(obj) >= 1:
            obj = obj[0]
        else:
            raise ValueError(f"Bad selection: {sele}")
        if group := pm.get_property("Group", obj):
            groups.append(group)
    polymers = [f"{g}.protein" for g in groups]
    assert len(polymers) > 0, "Please review your selections"
    
    ref_polymer = polymers[0]
    ref_sele = seles[0]
    site_sele = f"{ref_polymer} & ({ref_polymer} within {site_radius} of ({site}))"
    site_resis = []
    for at in pm.get_model(f"({site_sele}) & present & guide & polymer").atom:
        site_resis.append((at.model, at.index))
    
    if seq_align_omega:
        mapping = clustal_omega(polymers, omega_conservation.strip(), titles=seles)
    else:
        mapping = {}
        ref_model = pm.get_model(f"{ref_polymer} & present & guide & polymer")
        ref_map = []
        for at in ref_model.atom:
            ref_map.append(Residue(
                at.model, at.index, int(at.resi), at.chain,
                at.resn, RESN_TO_AA.get(at.resn, at.resn), ''
            ))
        mapping[ref_sele] = ref_map
        for poly, sele in zip(polymers, seles):
            if sele == ref_sele:
                continue
            current_model = pm.get_model(f"{poly} & present & guide & polymer")
            lookup = {(int(at.resi), at.chain): at for at in current_model.atom}
            current_map = []
            for ref_res in ref_map:
                match = lookup.get((int(ref_res.resi), ref_res.chain))
                if match:
                    at = match
                    current_map.append(
                        Residue(
                            at.model, at.index, int(at.resi), at.chain,
                            at.resn, RESN_TO_AA.get(at.resn, at.resn), ''
                        )
                    )
                else:
                    current_map.append(
                        Residue(None, -1, int(ref_res.resi), ref_res.chain, 'GAP', '-', ''))
            mapping[sele] = current_map

    ref_map = mapping[ref_sele]
    fpts = []
    for poly, (hs, map) in zip(polymers, mapping.items()):
        fpt = {}
        for ref_res, res in zip(ref_map, map):
            if (ref_polymer, ref_res.index) not in site_resis:
                continue
            lbl = (res.oneletter, res.conservation, res.resi, res.chain)
            cnt = pm.count_atoms(
                f"({hs}) within {contact_radius} of (byres %{poly} & index {res.index})"
            )
            fpt[lbl] = fpt.get(lbl, 0) + cnt
        fpts.append(fpt)
    
    if plot_fingerprints:
        fig, axs = plt.subplots(nrows=len(seles), ncols=1, sharex=sharex, constrained_layout=True)
        fig.supylabel('Atom Counts')
        if not isinstance(axs, (np.ndarray, list)):
            axs = [axs]
        if not all([len(fpts[0]) == len(fpt) for fpt in fpts]):
            raise ValueError(
                "All fingerprints must have the same length. "
                "Do you have incomplete structures?"
            )
        
        max_val = 0
        for ix, (ax, fpt, sele) in enumerate(zip(axs, fpts, seles)):
            labels = ["%s%s %s_%s" % k for k in fpt]
            if sharex and ix == 0:
                shared_labels = labels
            elif sharex and ix + 1 == len(seles):
                labels = shared_labels
            arange = np.arange(len(fpt))
            max_val = max(max(fpt.values()), max_val)
            ax.bar(arange, fpt.values(), color="C0")
            ax.set_title(sele)
            ax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
            if not sharex or sharex and ix + 1 == len(seles):
                ax.set_xticks(arange, labels=labels, rotation=90)
                ax.locator_params(axis="x", tight=True, nbins=nbins)
                for label in ax.xaxis.get_majorticklabels():
                    label.set_verticalalignment("top")
        if share_ylim:
            for ax in axs:
                ax.set_ylim(0, max_val * 1.05)
        
        if isinstance(plot_fingerprints, (str, Path)):
            fig.savefig(plot_fingerprints, dpi=300)
            if not quiet:
                print(f"Fingerprint figure saved to {plot_fingerprints}")
        
    corrs = []
    labels = []
    for i1, (fp1, sele1) in enumerate(zip(fpts, seles)):
        labels.append(sele1)
        for i2, (fp2, sele2) in enumerate(zip(fpts, seles)):
            if i1 >= i2:
                continue
            corr = pearsonr(list(fp1.values()), list(fp2.values())).statistic
            if np.isnan(corr):
                corr = 0
            corrs.append(1 - corr)
            if not quiet:
                print(f"Pearson correlation: {sele1} / {sele2}: {corr:.2f}")
    
    dendro, medoids = None, None
    if plot_hca:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=True)
        assert len(fpts) > 1, "Clustering requires multiple fingerprints, please add more selections."
        
        dendro, medoids = plot_hca_base(
            corrs,
            labels,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            only_medoids=only_medoids,
            annotate=annotate,
            axis=ax,
            vmin=0,
            vmax=2,
        )
        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment("right")
        if isinstance(plot_hca, (str, Path)):
            fig.savefig(plot_hca, dpi=300)
            if not quiet:
                print(f"Clusters figure saved to {plot_hca}")
    
    return fpts, corrs, dendro, medoids


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
    atoms1 = pm.get_coords(hs1)
    atoms2 = pm.get_coords(hs2)
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


class ResidueSimilarityMethod(StrEnum):
    JACCARD = "jaccard"
    OVERLAP = "overlap"


@new_command
def res_sim(
    hs1: Selection,
    hs2: Selection,
    radius: float = 4.0,
    seq_align: bool = False,
    method: ResidueSimilarityMethod = ResidueSimilarityMethod.JACCARD,
    quiet: bool = True,
):
    """
    Compute hotspots similarity by the Jaccard or overlap coefficient of nearby
    residues.

    OPTIONS
        hs1     hotspot 1
        hs2     hotspot 2
        radius  distance to consider residues near hotspots (default: 2)
        method  jaccard or overlap (default: jaccard)
        quiet   define verbosity

    EXAMPLES
        res_sim 8DSU.D_001*, 6XHM.D_001*
        res_sim 8DSU.CS_*, 6XHM.CS_*
    """
    group1 = pm.get_property("Group", hs1)  # FIXME it doesn't works with arbitrary objects
    group2 = pm.get_property("Group", hs2)

    sel1 = f"{group1}.protein within {radius} from ({hs1})"
    sel2 = f"{group2}.protein within {radius} from ({hs2})"

    resis1 = set()
    pm.iterate(sel1, "resis1.add((chain, resi))", space={"resis1": resis1})

    if group1 == group2 or not seq_align:
        resis2 = set()
        pm.iterate(sel2, "resis2.add((chain, resi))", space={"resis2": resis2})
    else:
        try:
            # FIXME Clustal Omega?
            aln_obj = pm.get_unused_name()
            pm.cealign(
                f"{group1}.protein", f"{group2}.protein", transform=0, object=aln_obj
            )
            raw = pm.get_raw_alignment(aln_obj)

            resis = {}
            pm.iterate(
                aln_obj, "resis[model, index] = (chain, resi)", space={"resis": resis}
            )

            site2 = [(a.chain, a.resi) for a in pm.get_model(sel2).atom]
            resis2 = set()
            for idx1, idx2 in raw:
                if resis[idx1] in site2:
                    resis2.add(resis[idx2])
        finally:
            pm.delete(aln_obj)

    try:
        match method:
            case ResidueSimilarityMethod.JACCARD:
                ret = len(resis1.intersection(resis2)) / len(resis1.union(resis2))
            case ResidueSimilarityMethod.OVERLAP:
                ret = len(resis1.intersection(resis2)) / min(len(resis1), len(resis2))
    except ZeroDivisionError:
        if not quiet:
            print("Your selection yields zero atoms.")
        return 0.0

    if not quiet:
        print(f"{method} similarity: {ret:.2}")
    return ret


class PairwiseFunction(StrEnum):
    HO = "ho"
    RESIDUE_JACCARD = "residue_jaccard"
    RESIDUE_OVERLAP = "residue_overlap"


@new_command
def plot_univariate_hca(
    sele: Selection,
    function: PairwiseFunction = PairwiseFunction.HO,
    radius: float = 2.0,
    align: bool = False,
    annotate: bool = False,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: float = 0.0,
    only_medoids: bool = False,
    plot: str = "",
    enable_heatmap: bool = False,
    rename_leafs: Optional[Dict[str, str]] = None
):
    """
    Compute the similarity between matching objects using a similarity function.

    OPTIONS
        objs        space separated list of object expressions
        method      ho, residue_jaccard, or residue_overlap (default: ho)
        radius      the radius to consider atoms in contact (default: 2.0)
        annotate    fill the cells with values

    EXAMPLES
        plot_pairwise_similarity *.D_000_*_*, function=residue_jaccard
        plot_pairwise_similarity *.D_*. align=True
        plot_pairwise_similarity *.D_000_*_* *.DS_*
    """

    
    objects = pm.get_object_list(sele)
    assert objects is not None and len(objects) >= 2, "At least two hotspots are required for comparison."

    X = []
    for idx1, obj1 in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if idx1 >= idx2:
                continue
            match function:
                case PairwiseFunction.HO:
                    ret = get_ho(obj1, obj2, radius=radius)
                case PairwiseFunction.RESIDUE_JACCARD:
                    ret = res_sim(
                        obj1,
                        obj2,
                        radius=radius,
                        method=ResidueSimilarityMethod.JACCARD,
                        seq_align=align,
                    )
                case PairwiseFunction.RESIDUE_OVERLAP:
                    ret = res_sim(
                        obj1,
                        obj2,
                        radius=radius,
                        method=ResidueSimilarityMethod.OVERLAP,
                        seq_align=align,
                    )
            X.append(1 - ret)
    dendro, medoids = plot_hca_base(X, objects, linkage_method, color_threshold, only_medoids, annotate, plot, vmin=0, vmax=1, enable_heatmap=enable_heatmap, rename_leafs=rename_leafs)
    return X, objects, dendro, medoids


class PrioritizationType(StrEnum):
    RESIDUE = "residue"
    ATOM = "atom"


@new_command
def hs_proj(
    sel: Selection,
    protein: Selection = "",
    radius: float = 4,
    type: PrioritizationType = PrioritizationType.RESIDUE,
    palette: str = "rainbow",
):
    """
    Colour atoms by proximity with FTMap probes.

    OPTIONS:
        sel         probes selection.
        protein     object which will be coloured.
        max_dist    maximum distance in Angstroms (default: 4).
        type        residue or type (default: residue).
        palette     spectrum colour palette (default: rainbow).
    """

    if not protein:
        group = sel.split(".", maxsplit=1)[0]
        protein = f"{group}.protein"

    pm.alter(protein, "q=0")
    for prot_atom in pm.get_model(f"({protein}) within {radius} of ({sel})").atom:
        match type:
            case PrioritizationType.RESIDUE:
                prot_atom_sel = f"byres index {prot_atom.index}"
            case PrioritizationType.ATOM:
                prot_atom_sel = f"index {prot_atom.index}"
        count = count_molecules(f"({sel}) within {radius} of ({prot_atom_sel})")
        pm.alter(prot_atom_sel, f"q={count}")

    pm.hide("everything", protein)
    match type:
        case PrioritizationType.RESIDUE:
            pm.show("surface", protein)
        case PrioritizationType.ATOM:
            pm.show("cartoon", protein)
            pm.show("sticks", "q>0")
    pm.spectrum("q", palette=palette, selection=protein)


class DistanceMethod(StrEnum):
    EUCLIDEAN = "euclidean"
    CITYBLOCK = "cityblock"
    DICE = "dice"


@new_command
def plot_multivariate_hca(
    exprs: Selection,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: float = 0.0,
    only_medoids: bool = False,
    annotate: bool = False,
    plot: str = None,
    dist_method: DistanceMethod = DistanceMethod.EUCLIDEAN,
    enable_heatmap: bool = False,
    rename_leafs: Optional[Dict[str, str]] = None
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
        n_props = 4
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
            CD = pm.get_property("CD", obj)
            MD = pm.get_property("MD", obj)
            p[ix, :] = np.array([ST, S0, CD, MD, x, y, z])
        elif hs_type == "CS":
            ST = pm.get_property("ST", obj)
            p[ix, :] = np.array([ST, x, y, z])
        elif hs_type == "ACS":
            ST = pm.get_property("ST", obj)
            MD = pm.get_property("MD", obj)
            p[ix, :] = np.array([ST, MD, x, y, z])

    
    p = (p - p.mean(axis=0)) / (p.std(axis=0) + 1e-8)
    X = distance.pdist(p, dist_method)
    dendro, medoids = plot_hca_base(X, labels, linkage_method, color_threshold, only_medoids, annotate, plot, enable_heatmap=enable_heatmap, rename_leafs=rename_leafs)
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


def calc_medchem_bind_metrics(lig_sele, pki):
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

def plot_overlap_activity_scatter(hs_sele, sele_b, function, radius, annotate, bind_df):
    if len(pm.get_object_list(hs_sele)) != 1:
        raise ValueError("Only one hotspot can be analyzed at time.")
    overlap_df = plot_overlap_matrix(
        sele_a=hs_sele,
        sele_b=sele_b,
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)
    for ax, metric in [
        (ax1, 'PKI'),
        (ax2, 'LE'),
        (ax3, 'BEI'),
        (ax4, 'FQ'),
    ]:
        for label, subset_df in df.groupby('A'):
            x = subset_df[function_col] / subset_df[function_col].iloc[ix_frag]
            y = subset_df[metric] / subset_df[metric].iloc[ix_frag]
            ax.scatter(x, y, label=label)
            rows = zip(x, y, subset_df['Ligand'], subset_df['Label'])
            for x, y, obj, label in rows:
                s = label.strip() or obj
                ax.text(x, y, s)
        ax.set_xlabel(function_col)
        ax.set_ylabel(metric)
    fig.legend(
        loc='lower center',
        bbox_to_anchor=(1, 1.02),
        fontsize='small',
        ncol=3,
    )
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

        self.allowNested = QCheckBox()
        self.allowNested.setChecked(False)
        boxLayout.addRow("Allow nested hotspots", self.allowNested)

        self.maxCollisions = QDoubleSpinBox()
        self.maxCollisions.setRange(0.0, 1.0)
        self.maxCollisions.setSingleStep(0.05)
        self.maxCollisions.setValue(0.25)
        boxLayout.addRow("Max collisions", self.maxCollisions)
        
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
        allow_nested = self.allowNested.isChecked()
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

        def hideEvent(self, evt):
            self.clearSelection()

    def __init__(self):
        super().__init__()
        self.selected_objs = set()
        self.current_tab = "HS"

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.filter_line = QLineEdit("*")
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
                "S1",
                "CD",
                "MD",
                "Length",
            ],
            ("CS", "CS"): ["ST"],
        }
        if XDRUGPY_EXPERIMENTAL_VERSION:
            self.hotspotsMap[("Hotspots", "HS")].insert(4, "SZ")
            self.hotspotsMap[("Hotspots", "HS")].append("isComplex")
            self.hotspotsMap[("ACS", "ACS")] = ["Class", "ST", "MD"]

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

        self.hotspotSeleLine = QLineEdit("*")
        mainLayout.addWidget(self.hotspotSeleLine)

        tab = QTabWidget()
        mainLayout.addWidget(tab)

        groupBox = QWidget()
        mainLayout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)
        tab.addTab(groupBox, "Parameters")

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

        self.enableHeatmapCheck = QCheckBox()
        boxLayout.addRow("Heatmap:", self.enableHeatmapCheck)

        self.annotateCheck = QCheckBox()
        boxLayout.addRow("Annotate:", self.annotateCheck)

        self.onlyMedoidsCheck = QCheckBox()
        self.onlyMedoidsCheck.setChecked(False)
        boxLayout.addRow("Show only medoids:", self.onlyMedoidsCheck)

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
            row = self.table.rowCount()
            if item.row() != row-1:
                return
            if self.table.item(row-1, 0) is None:
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(""))
                self.table.setItem(row, 1, QTableWidgetItem(""))
                return
            
            col0_text = self.table.item(row-1, 0)
            col0_text = col0_text.text().strip()
            
            col1_text = self.table.item(row-1, 1)
            col1_text = col1_text.text().strip()

            if (col0_text == "" and col1_text == ""):
                return
            
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(""))
            self.table.setItem(row, 1, QTableWidgetItem(""))
            
            self.table.blockSignals(False)

        layout = QHBoxLayout()
        mainLayout.addLayout(layout)

        groupBox = QGroupBox("Univariate analysis")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.univariateFunctionCombo = QComboBox()
        self.univariateFunctionCombo.addItems([e.value for e in PairwiseFunction])
        boxLayout.addRow("Function:", self.univariateFunctionCombo)

        self.radiusSpin = QDoubleSpinBox()
        self.radiusSpin.setValue(4)
        self.radiusSpin.setSingleStep(0.5)
        self.radiusSpin.setDecimals(2)
        self.radiusSpin.setMinimum(1)
        self.radiusSpin.setMaximum(10)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.pairwiseSeqAlignCheck = QCheckBox()
        self.pairwiseSeqAlignCheck.setChecked(False)
        boxLayout.addRow("Sequence align:", self.pairwiseSeqAlignCheck)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_univariate)
        boxLayout.addWidget(plotButton)

        groupBox = QGroupBox("Multivariate analysis")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.multivariateFunctionCombo = QComboBox()
        self.multivariateFunctionCombo.addItems([e.value for e in DistanceMethod])
        boxLayout.addRow("Distance:", self.multivariateFunctionCombo)

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
    
    def plot_univariate(self):
        sele = self.hotspotSeleLine.text()
        function = self.univariateFunctionCombo.currentText()
        radius = self.radiusSpin.value()
        align = self.pairwiseSeqAlignCheck.isChecked()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        only_medoids = self.onlyMedoidsCheck.isChecked()
        annotate = self.annotateCheck.isChecked()
        enable_heatmap = self.enableHeatmapCheck.isChecked()

        plot_univariate_hca(
            sele=sele,
            function=function,
            radius=radius,
            align=align,
            annotate=annotate,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            only_medoids=only_medoids,
            enable_heatmap=enable_heatmap,
            rename_leafs=self.getLeafLabels(),
        )

    def plot_multivariate_hca(self):
        sele = self.hotspotSeleLine.text()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        only_medoids = self.onlyMedoidsCheck.isChecked()
        annotate = self.annotateCheck.isChecked()
        dist_method = self.multivariateFunctionCombo.currentText()
        enable_heatmap = self.enableHeatmapCheck.isChecked()

        plot_multivariate_hca(
            exprs=sele,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            only_medoids=only_medoids,
            annotate=annotate,
            dist_method=dist_method,
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
        while self.rowCount() > 0:
            self.removeRow(0)
        for obj in objects:
            self.insertRow(self.rowCount())
            line = self.rowCount() - 1

            item = QTableWidgetItem(obj)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.setItem(line, 0, item)

            self.setItem(line, 1, QTableWidgetItem(""))
            for ix in range(2, 7):
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(line, ix, item)
            
            self.setItem(line, 7, QTableWidgetItem(""))
                
        self.setSortingEnabled(True)
    
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
            if not pki:
                continue
            le = self.item(r, 2).text()
            bei = self.item(r, 3).text()
            fq = self.item(r, 4).text()
            ha = self.item(r, 5).text()
            mw = self.item(r, 6).text()
            label = self.item(r, 7).text()
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


class OverlapWidget(QWidget):

    def __init__(self):
        super().__init__()
        
        layout = QFormLayout()
        self.setLayout(layout)

        seleContainer = QWidget()
        seleLayout = QHBoxLayout(seleContainer)
        seleLayout.setContentsMargins(0, 0, 0, 0)
        
        layout.addRow("Selections:", seleContainer)
        
        self.aSeleLine = QLineEdit()
        self.aSeleLine.setPlaceholderText("Selection A...")
        seleLayout.addWidget(self.aSeleLine)
        self.aSeleLine.textChanged.connect(self.validateUpdateWidget)

        self.bSeleLine = QLineEdit()
        self.bSeleLine.setPlaceholderText("Selection B...")
        self.bSeleLine.textChanged.connect(self.validateUpdateWidget)
        seleLayout.addWidget(self.bSeleLine)
        
        self.functionCombo = QComboBox()
        self.functionCombo.addItems([e.value for e in OverlapFunction])
        layout.addRow("Function:", self.functionCombo)

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

        self.table = LigandTableWidget()
        self.table.setEnabled(False)
        layout.addRow(self.table)
        @self.table.itemChanged.connect
        def itemChanged(item):
            self.table.setSortingEnabled(False)
            row = item.row()

            if item.column() == 1:
                if item.text() == "":
                    le = ""
                    bei = ""
                    fq = ""
                    ha = ""
                    mw = ""
                else:
                    lig_obj = self.table.item(row, 0).text()
                    pki = float(self.table.item(row, 1).text())
                    bind = calc_medchem_bind_metrics(lig_obj, pki)
                    le = bind['le']
                    bei = bind['bei']
                    fq = bind['fq']
                    ha = bind['ha']
                    mw = bind['mw']
                
                if self.table.item(row, 2):
                    self.table.item(row, 2).setText(f"{le:.3f}")
                    self.table.item(row, 3).setText(f"{bei:.3f}")
                    self.table.item(row, 4).setText(f"{fq:.3f}")
                    self.table.item(row, 5).setText(f"{ha}")
                    self.table.item(row, 6).setText(f"{mw:.2f}")
            self.table.setSortingEnabled(True)

        self.container2 = QWidget()
        hLayout2 = QHBoxLayout(self.container2)
        layout.addRow(self.container2)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_overlap_activity_scatter)
        hLayout2.addWidget(plotButton)

        exportButton = QPushButton("Export")
        exportButton.clicked.connect(self.export_ligand_data)
        hLayout2.addWidget(exportButton)

    def validateUpdateWidget(self):
        a_sele = self.aSeleLine.text().strip()
        b_ph = "Selection B..."
        if a_sele != "":
            b_ph = a_sele
        self.bSeleLine.setPlaceholderText(b_ph)
        objs_a = pm.get_object_list(a_sele)
        if objs_a is None:
            objs_a = []
        if len(objs_a) == 0:
            self.container1.setEnabled(False)
        else:
            self.container1.setEnabled(True)

        b_sele = self.bSeleLine.text().strip()
        objs_b = pm.get_object_list(b_sele)
        if objs_b is None:
            objs_b = []
        self.table.refresh(objs_b)
        if len(objs_b) > 0:
            self.table.setEnabled(True)
            self.container2.children()[2].setEnabled(True)
            if len(objs_a) == 1:
                self.container2.children()[1].setEnabled(True)
        else:
            self.table.setEnabled(False)
            self.container2.children()[1].setEnabled(False)
            self.container2.children()[2].setEnabled(False)
        if len(objs_a) != 1:
            self.container2.children()[1].setEnabled(False)
            
    
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

    def plot_overlap_activity_scatter(self):
        sele_a = self.aSeleLine.text().strip()
        sele_b = self.bSeleLine.text().strip()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        annotate = self.annotateCheck.isChecked()

        bind_df = self.table.getDataFrame()
        bind_df = bind_df.set_index('Ligand')

        plot_overlap_activity_scatter(
            hs_sele=sele_a,
            sele_b=sele_b,
            function=function,
            radius=radius,
            annotate=annotate,
            bind_df=bind_df
        )
        plt.show()
    
    def export_ligand_data(self):
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

class CountWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        self.setLayout(layout)

        groupBox = QGroupBox("Color projection")
        layout.addWidget(groupBox)
        projBox = QFormLayout()
        groupBox.setLayout(projBox)

        self.multiSeleLine = QLineEdit()
        projBox.addRow("Expressions:", self.multiSeleLine)

        self.proteinExpressionLine = QLineEdit()
        projBox.addRow("Protein:", self.proteinExpressionLine)

        self.radiusSpin = QDoubleSpinBox()
        self.radiusSpin.setValue(4)
        self.radiusSpin.setDecimals(2)
        self.radiusSpin.setSingleStep(0.5)
        self.radiusSpin.setMinimum(2)
        self.radiusSpin.setMaximum(10)
        projBox.addRow("Contact radius:", self.radiusSpin)

        self.typeCombo = QComboBox()
        self.typeCombo.addItems([e.value for e in PrioritizationType])
        projBox.addRow("Type:", self.typeCombo)

        self.paletteLine = QLineEdit("rainbow")
        projBox.addRow("Palette:", self.paletteLine)

        drawButton = QPushButton("Draw")
        drawButton.clicked.connect(self.draw_projection)
        projBox.addWidget(drawButton)

        fptBox = QGroupBox("Fingerprint vector")
        layout.addWidget(fptBox)

        fptLayout = QVBoxLayout(fptBox)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        fptLayout.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        
        scrollLayout = QFormLayout(container)
        

        self.multiSelesLine = QLineEdit("")
        scrollLayout.addRow("Multi sele:", self.multiSelesLine)

        self.siteSelectionLine = QLineEdit("*")
        scrollLayout.addRow("Focus site:", self.siteSelectionLine)

        self.siteRadiusSpin = QDoubleSpinBox()
        self.siteRadiusSpin.setValue(5)
        self.siteRadiusSpin.setDecimals(1)
        self.siteRadiusSpin.setSingleStep(1)
        self.siteRadiusSpin.setMinimum(0)
        self.siteRadiusSpin.setMaximum(10)
        scrollLayout.addRow("Site radius:", self.siteRadiusSpin)

        self.contactRadiusSpin = QDoubleSpinBox()
        self.contactRadiusSpin.setValue(4)
        self.contactRadiusSpin.setDecimals(1)
        self.contactRadiusSpin.setSingleStep(0.5)
        self.contactRadiusSpin.setMinimum(3)
        self.contactRadiusSpin.setMaximum(6)
        scrollLayout.addRow("Contact radius:", self.contactRadiusSpin)

        self.omegaCheck = QCheckBox()
        self.omegaCheck.setChecked(False)
        scrollLayout.addRow("Clustal Omega:", self.omegaCheck)

        @self.omegaCheck.stateChanged.connect
        def stateChanged(checkState):
            if checkState == QtCore.Qt.Checked:
                omegaBox.setEnabled(True)
            else:
                omegaBox.setEnabled(False)

        omegaBox = QGroupBox()
        scrollLayout.addRow(omegaBox)
        scrollLayout.setWidget(scrollLayout.rowCount(), QFormLayout.SpanningRole, omegaBox)
        omegaLayout = QFormLayout()
        omegaBox.setLayout(omegaLayout)
        omegaBox.setEnabled(False)

        self.omegaConservation = QLineEdit()
        self.omegaConservation.setText("*:.")
        omegaLayout.addRow("Conservation symbols:", self.omegaConservation)

        self.fingerprintsCheck = QCheckBox()
        self.fingerprintsCheck.setChecked(False)
        scrollLayout.addRow("Fingerprints:", self.fingerprintsCheck)
        @self.fingerprintsCheck.stateChanged.connect
        def stateChanged(checkState):
            if checkState == QtCore.Qt.Checked:
                fptBox.setEnabled(True)
            else:
                fptBox.setEnabled(False)

        fptBox = QGroupBox()
        scrollLayout.addRow(fptBox)
        scrollLayout.setWidget(scrollLayout.rowCount(), QFormLayout.SpanningRole, fptBox)
        fptLayout = QFormLayout()
        fptBox.setLayout(fptLayout)
        fptBox.setEnabled(False)

        self.nBinsSpin = QSpinBox()
        self.nBinsSpin.setValue(20)
        self.nBinsSpin.setMinimum(0)
        self.nBinsSpin.setMaximum(500)
        fptLayout.addRow("Num bins:", self.nBinsSpin)

        self.shareYLimCheck = QCheckBox()
        self.shareYLimCheck.setChecked(True)
        fptLayout.addRow("Share y limit:", self.shareYLimCheck)

        self.sharexCheck = QCheckBox()
        self.sharexCheck.setChecked(True)
        fptLayout.addRow("Share x axis:", self.sharexCheck)

        self.hcaCheck = QCheckBox()
        self.hcaCheck.setChecked(False)

        scrollLayout.addRow("Clustering:", self.hcaCheck)
        @self.hcaCheck.stateChanged.connect
        def stateChanged(checkState):
            if checkState == QtCore.Qt.Checked:
                hcaBox.setEnabled(True)
            else:
                hcaBox.setEnabled(False)

        hcaBox = QGroupBox()
        scrollLayout.addRow(hcaBox)
        scrollLayout.setWidget(scrollLayout.rowCount(), QFormLayout.SpanningRole, hcaBox)
        hcaLayout = QFormLayout()
        hcaBox.setLayout(hcaLayout)
        hcaBox.setEnabled(False)

        self.annotateCheck = QCheckBox()
        self.annotateCheck.setChecked(True)
        hcaLayout.addRow("Annotate:", self.annotateCheck)

        self.linkageMethodCombo = QComboBox()
        self.linkageMethodCombo.addItems([e.value for e in LinkageMethod])
        hcaLayout.addRow("Linkage:", self.linkageMethodCombo)

        self.colorThresholdSpin = QDoubleSpinBox()
        self.colorThresholdSpin.setMinimum(0)
        self.colorThresholdSpin.setMaximum(10)
        self.colorThresholdSpin.setValue(0)
        self.colorThresholdSpin.setSingleStep(0.1)
        self.colorThresholdSpin.setDecimals(2)
        hcaLayout.addRow("Color threshold:", self.colorThresholdSpin)

        self.onlyMedoidsCheck = QCheckBox()
        self.onlyMedoidsCheck.setChecked(False)
        hcaLayout.addRow("Show only medoids:", self.onlyMedoidsCheck)
        
        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_fingerprint)
        scrollLayout.addWidget(plotButton)

    def draw_projection(self):
        hotspots = self.multiSeleLine.text()
        protein = self.proteinExpressionLine.text()
        radius = self.radiusSpin.value()
        type = self.typeCombo.currentText()
        palette = self.paletteLine.text()

        hs_proj(hotspots, protein, radius, type, palette)

    def plot_fingerprint(self):
        multi_seles = self.multiSelesLine.text()
        site = self.siteSelectionLine.text()
        site_radius = self.siteRadiusSpin.value()
        contact_radius = self.contactRadiusSpin.value()
        plot_fingerprints = self.fingerprintsCheck.isChecked()
        plot_hca = self.hcaCheck.isChecked()
        seq_align_omega = self.omegaCheck.isChecked()
        omega_conservation = self.omegaConservation.text().strip()
        nbins = self.nBinsSpin.value()
        share_ylim = self.shareYLimCheck.isChecked()
        sharex = self.sharexCheck.isChecked()
        annotate = self.annotateCheck.isChecked()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        only_medoids = self.onlyMedoidsCheck.isChecked()

        fpt_sim(
            multi_seles,
            site,
            site_radius,
            contact_radius=contact_radius,
            plot_fingerprints=plot_fingerprints,
            plot_hca=plot_hca,
            seq_align_omega=seq_align_omega,
            omega_conservation=omega_conservation,
            nbins=nbins,
            share_ylim=share_ylim,
            sharex=sharex,
            annotate=annotate,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            only_medoids=only_medoids,
        )
        plt.show()


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
        tab.addTab(HcaWidget(), "HCA")
        tab.addTab(OverlapWidget(), "Overlap")
        if XDRUGPY_EXPERIMENTAL_VERSION:
            tab.addTab(CountWidget(), "Fingerprints")

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
