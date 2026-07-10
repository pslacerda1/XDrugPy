from __future__ import annotations

import os.path
import re
from types import SimpleNamespace
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Literal, List, Dict, Tuple
from functools import lru_cache
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix, distance, cKDTree
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib import pyplot as plt, axes
from strenum import StrEnum
import networkx as nx
import pymol
from pymol import cmd as pm
from pymol.exporting import _resn_to_aa as RESN_TO_AA
from pymol_new_command import new_command

from .utils import (
    Selection,
    plot_hca_base,
    clustal_omega,
    AligMethod
)


@dataclass
class ECluster:
    selection: str
    probe_type: str
    coords: Any = field(repr=False, hash=False)
    idx: int
    ST: int

def objects_from_kvfinder(group: str, kvfound: List[str]) -> List[str]:
    cavities = []
    for cavity, residues in kvfound.items():
        cavity = f'{group}.KV.{cavity}'
        if len(residues) >= 2:
            resi, chain, _ = residues[0]
            pm.select(cavity, f'(i. {resi} AND c. {chain})')
        else:
            raise NotImplemented("Not supposed to a cavity spans only 1 or zero residues.")
        for resi, chain, resn in residues[1:]:
            pm.select(cavity, f'{cavity} OR (i. {resi} AND c. {chain})')
        pm.select(cavity, f'{cavity} AND {group}.protein')
        pm.disable(cavity)
        cavities.append(cavity)
    return cavities

def kvfinder_constitutional_from_pdb_string(pdbstr: str):
    # hello GPT
    import tempfile
    from pyKVFinder.grid import get_vertices, detect, constitutional
    from pyKVFinder.utils import read_vdw, read_pdb, VDW

    step = 0.6
    probe_in = 1.4
    probe_out = 4.0
    removal_distance = 2.4
    volume_cutoff = 5.0
    ligand_cutoff = 5.0
    surface = "SES"
    ignore_backbone = False
    model = None
    nthreads = None
    verbose = False

    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=True) as tmp:
        pdbstr = (
            '\n'
            .join(
                l for l
                in pdbstr.splitlines()
                if l.startswith('ATOM')
            )
        )
        tmp.write(pdbstr)
        tmp.flush()
        atomic = read_pdb(tmp.name, read_vdw(VDW), model)

        vertices = get_vertices(atomic, probe_out, step)
        ncav, cavities = detect(
            atomic,
            vertices,
            step,
            probe_in,
            probe_out,
            removal_distance,
            volume_cutoff,
            None,
            ligand_cutoff,
            False,
            surface,
            nthreads,
            verbose,
        )

        if ncav <= 0:
            return {}

        residues = constitutional(
            cavities,
            atomic,
            vertices,
            step,
            probe_in,
            ignore_backbone,
            None,
            nthreads,
            verbose,
        )
    return residues


def get_coords(sel, state=1):
    return pm.get_coords(sel, state)

# def process_eclusters(group, eclusters):
#     for acs in eclusters:
#         new_name = f"{group}.ACS.{acs.probe_type}.{acs.idx:02}"
#         pm.set_name(acs.selection, new_name)
#         acs.selection = new_name
#         pm.group(group, new_name)

#         coords = pm.get_coordset(new_name)
#         md = distance_matrix(coords, coords).max()

#         set_properties(
#             acs,
#             new_name,
#             {
#                 "Type": "ACS",
#                 "Group": group,
#                 "Selection": new_name,
#                 "Class": acs.probe_type,
#                 "ST": acs.ST,
#                 "MD": round(md, 2),
#             },
#         )
#     pm.delete("clust.*")


def set_properties(obj_name, properties):
    for prop, value in properties.items():
        pm.set_property(prop, value, obj_name)
        pm.set_atom_property(prop, value, obj_name)


def parse_pdb_string(
        pdbstr: str,
        cavities: List[str]
) -> Tuple[List[Cluster], List[Hotspot]]:

    clusters = []
    hotspots = []
    remark_re = re.compile(r'([a-zA-Z0-9_]+)=([a-zA-Z0-9._]+)')
    for line in pdbstr.split('\n'):
        
        if not line.startswith('REMARK'):
            continue
        
        d = {}
        for m in remark_re.finditer(line[7:]):
            k = m.group(1)
            v = m.group(2)
            d[k] = v
        if not d:
            continue

        if '.CS.' in d['Object']:
            cs = Cluster(
                Group=d['Group'],
                Object=d['Object'],
                Coords=pm.get_coordset(d['Object']),
                S=int(d['S']),
            )
            cs.save_into_properties()
            clusters.append(cs)

        elif not d['Object'].endswith('.protein'):
            hs = Hotspot(
                Group=d['Group'],
                Object=d['Object'],
                Coords=pm.get_coordset(d['Object']),
                Class=d['Class'],
                ST=int(d['ST']),
                S0=int(d['S0']),
                S1=int(d['S1']),
                SZ=int(d['SZ']),
                CD=float(d['CD']),
                MD=float(d['MD']),
                Length=int(d['Len']),
                Kavity=None,
            )
            
            max_touch = 0
            max_cavity = None
            for cavity in cavities:
                n_atoms = pm.count_atoms(hs.Object)
                in_touch = pm.count_atoms(f"{hs.Object} NEAR_TO 4 OF {cavity}")
                if in_touch > max_touch and in_touch > 0.80 * n_atoms:
                    max_touch = in_touch    
                    max_cavity = cavity

            hs.Kavity = max_cavity
            hs.save_into_properties()
            hotspots.append(hs)

    return clusters, hotspots


@dataclass
class Cluster:
    Group: str
    Object: str
    S: int

    Coords: Any = field(repr=False, hash=False)
    Type: Literal["CS"] =  field(default="CS", repr=False)

    def save_into_properties(self):
        d = asdict(self)
        del d['Coords']
        set_properties(self.Object, d)
    
    @classmethod
    def from_object_name(cls, name: str) -> Cluster:
        assert ".CS." in name
        assert "CS" in pm.get_property("Type", name)
        assert name == pm.get_property("Object", name)

        cluster = Cluster(
            Group=pm.get_property('Group', name),
            Object=name,
            S=pm.get_property('S', name),
            Coords=pm.get_coordset(name),
        )
        return cluster


@dataclass
class Hotspot:
    Group: str
    Object: str
    
    Class: Literal["D", "DS", "DL", "B", "BS", "BL", None]
    ST: int
    S0: int
    S1: int
    SZ: int
    CD: float
    MD: float
    Length: int
    Kavity: str | None

    Coords: Any = field(repr=False, hash=False)
    Type: Literal["HS"] =  field(default="HS", repr=False)

    def save_into_properties(self):
        d = asdict(self)
        del d['Coords']
        set_properties(self.Object, d)

    @classmethod
    def from_object_name(cls, name: str) -> Hotspot:
        assert "HS" == pm.get_property("Type", name)
        assert name == pm.get_property("Object", name)
        hs = Hotspot(
            Group=pm.get_property('Group', name),
            Object=name,
            Coords=pm.get_coordset(name),
            Class=pm.get_property('Class', name),
            ST=pm.get_property('ST', name),
            S0=pm.get_property('S0', name),
            S1=pm.get_property('S1', name),
            SZ=pm.get_property('SZ', name),
            CD=pm.get_property('CD', name),
            MD=pm.get_property('MD', name),
            Length=pm.get_property('Length', name),
            Kavity=pm.get_property('Kavity', name)
        )
        return hs


def load_ftmap(
    filename: Path | str,
    group: Optional[str] = None,
    deep_search: bool = True,
    remove_nested: bool = True,
    clash_threshold: float = 0.15,
    pretty: bool = False,
):
    """
    DESCRIPTION

        Loads FTMap (or FTMove) results into PyMOL, organizes them into a structured
        hierarchy, and calculates binding hotspots.

        This command automates the identification, classification and visualization of
        FTMap probes and hotspots,  assigning colors by class type and grouping objects
        logically within the PyMOL object menu.

    ARGUMENTS

        filename:
            Path to the FTMap .pdb file.

        group:
            The name of the top-level PyMOL group. If not provided, the 
            basename of the file is used.

        deep_search:
            Determines if hotspots can overlap/nest within larger 
            defined volumes.

        remove_nested:
            Remove hotspots made that are fully nested/inside other.

        clash_threshold:
            The tolerance percentage for steric clashes allowed when
            defining hotspot connectivity.
        
        pretty:
            Enable beautiful visualizations of the loaded FTMap results. Also enable
            grouping of hotspots by type.

    EXAMPLES

        load_ftmap 1w9h_ftmap.pdb
        load_ftmap results.pdb, my_protein, max_collisions=0.20
    """
    try:
        pm.set('defer_updates', 1)
        try:
            return _load_ftmap(
                filename=filename,
                group=group,
                deep_search=deep_search,
                clash_threshold=clash_threshold,
                remove_nested=remove_nested,
                pretty=pretty,
            )
        except:
            return _load_ftmap(
                filename=filename,
                group=group,
                deep_search=deep_search,
                remove_nested=remove_nested,
                clash_threshold=clash_threshold,
                pretty=pretty,
            )
    finally:
        pm.set('defer_updates', 0)


def _load_ftmap(
    filename: Path,
    group: str = "",
    deep_search: bool = True,
    remove_nested=True,
    clash_threshold: float = 0.15,
    pretty: bool = False,
):
    if not group:
        group = os.path.splitext(os.path.basename(str(filename)))[0]
    group = pm.get_legal_name(group)

    cmd = [
        'xdrugpy_hotspot_finder',
        '-g', group,
        '--input', str(filename),
        '--clash-threshold', str(clash_threshold),
    ]
    if deep_search:
        cmd.append('--deep-search')
    if remove_nested:
        cmd.append('--remove-nested')
    
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"xdrugpy_hotspot_finder failed with exit code {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    pdbstr = proc.stdout
    pm.read_pdbstr(pdbstr, group)
    for klass in ['D', 'DS', 'DL', 'B', 'BS', 'BL', 'CS']:
        pm.disable(f"{group}.{klass}")
    
    kvfound = kvfinder_constitutional_from_pdb_string(pdbstr)
    cavities = objects_from_kvfinder(group, kvfound)
    clusters, hotspots = parse_pdb_string(pdbstr, cavities)

    pm.group(group, f"{group}.*")


    if pretty:
        pm.group(group, f"{group}.protein")
        
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

        pm.order(f"{group}.BS", location="top")
        pm.order(f"{group}.BL", location="top")
        pm.order(f"{group}.B", location="top")
        pm.order(f"{group}.DS", location="top")
        pm.order(f"{group}.DL", location="top")
        pm.order(f"{group}.D", location="top")

        pm.order(f"{group}.KV.*", location="top")
        pm.order(f"{group}.CS.*", location="top")
        pm.order(f"{group}.protein", location="top")

        pm.group(f"{group}.CS", f"{group}.CS.*")
        pm.group(f"{group}.KV", f"{group}.KV.*")

        pm.group(f"{group}.D", f"{group}.D.*")
        pm.group(f"{group}.B", f"{group}.B.*")
        pm.group(f"{group}.DS", f"{group}.DS.*")
        pm.group(f"{group}.BS", f"{group}.BS.*")
        pm.group(f"{group}.DL", f"{group}.DL.*")
        pm.group(f"{group}.BL", f"{group}.BL.*")

        pm.group(group, f"{group}.protein")
        pm.group(group, f"{group}.KS")
        pm.group(group, f"{group}.CS")
        pm.group(group, f"{group}.D")
        pm.group(group, f"{group}.B")
        pm.group(group, f"{group}.DS")
        pm.group(group, f"{group}.BS")
        pm.group(group, f"{group}.DL")
        pm.group(group, f"{group}.BL")

        pm.set("mesh_mode", 1)
        pm.orient("all")

        # groups = [
        #     name
        #     for name in pm.get_names()
        #     if pm.get_type(name) == 'object:group'
        #         and name.startswith(f"{group}.")
        #         and pm.count_atoms(f"%{name}") == 0
        # ]
        # for grp in groups:
        #     if grp in pm.get_names():
        #         pm.delete(grp)
    
    return SimpleNamespace(
        clusters=clusters,
        hotspots=hotspots
    )

    
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
    DESCRIPTION

        Calculates the Fractional Overlap (FO) between two selections. 
        FO is defined as the fraction of atoms in sel1 that are within 
        a specified radius of any atom in sel2.

    ARGUMENTS

        sel1:
            The 'query' selection. The FO is normalized by the number 
            of atoms in this selection.

        sel2:
            The 'target' selection used as the proximity reference.

        radius:
            The distance cutoff in Angstroms to consider an atom 'overlapping'.

    NOTES

        Value ranges from 0.0 to 1.0. An FO of 1.0 means every atom in sel1 
        is within the radius of sel2.
    """
    if isinstance(sel1, np.ndarray):
        xyz1 = sel1
    else:
        xyz1 = get_coords(sel1, state=state1)
    if isinstance(sel2, np.ndarray):
        xyz2 = sel2
    else:
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
    DESCRIPTION

        Calculates Density Correlation, i.e. the total number of
        pairwise atomic contacts between two selections based on
        a distance threshold.
        
    ARGUMENTS

        sel1, sel2:
            The two groups of atoms to check for proximity.

        radius:
            Distance cutoff.

    RETURNS

        The total count of atom-atom pair count within the radius.
    """
    if isinstance(sel1, np.ndarray):
        xyz1 = sel1
    else:
        xyz1 = get_coords(sel1, state=state1)
    if isinstance(sel2, np.ndarray):
        xyz2 = sel2
    else:
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
    DESCRIPTION

        Calculates the Density Correlation Efficiency (DCE). 
        This is the total number of contacts (DC) normalized by the 
        total number of atoms in the first selection.

    ARGUMENTS

        sel1, sel2:
            The two groups of atoms to check for proximity.

        radius:
            Distance cutoff.
            
    NOTES

        DCE = get_dc(sel1, sel2) / count_atoms(sel1)
        This is useful for comparing binding efficiency across ligands 
        of different sizes.
    """
    if isinstance(sel1, np.ndarray):
        xyz1 = sel1
    else:
        xyz1 = get_coords(sel1, state=state1)
    dce = get_dc(
        xyz1,
        sel2,
        radius=radius,
        state1=state1,
        state2=state2
    ) / len(xyz1)
    if not quiet:
        print(f"DCE: {dce:.2f}")
    return dce


class LinkageMethod(StrEnum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"


class HcaOverlapFunction(StrEnum):
    FO_AVG = "fo_avg"
    JACCARD = "jaccard"
    OVERLAP = "overlap"


@new_command
def calc_univariate_hca(
    sele: Selection,
    overlap_function: HcaOverlapFunction.FO_AVG = HcaOverlapFunction.FO_AVG,
    radius: float = 2.0,
    linkage_method: LinkageMethod = LinkageMethod.WARD,
    color_threshold: float = -1.0,
    nclusters: int = -1,
    only_medoids: bool = False,
    annotate: bool = False,
    rename_leafs: Optional[Dict[str, str]] = None,
    figure_title: str | None = None,
    dendrogram_plot: str = '',
    heatmap_plot: str = '',
):
    """
    DESCRIPTION
        Performs an Univariate Hierarchical Analysis (HCA) on consensus sites
        or hotspot.

    ARGUMENTS

        exprs:
            A PyMOL selection containing the objects to compare. All objects must
            be of the same type. All hotspots or all consensus sites.

        overlap_function:
            The overlap function to measure the similarity between two objects.

        seq_align_before_overlap:
            Do alignment before comparing the polymer residues near the two objects.
            
        linkage_method:
            The clustering algorithm for the dendrogram. 

        radius: float
            The distance cutoff (Angstroms) passed to the overlap function.

        color_threshold:
            Distance cutoff for cluster dendrogram branches. Disabled by default.

        ncluters:
            Cutoff only a number of cluster dendrogram branchs. Disabled by default.

        only_medoids:
            If True, focuses analysis or visualization only on the cluster medoids.
            
        annotate:
            If True, writes the numerical values directly inside the heatmap 
            cells. Text color (black/white) is automatically adjusted for 
            legibility based on the cell intensity.

        rename_leafs:
            A dictionary mapping PyMOL object names to user-friendly labels.

    RETURNS
        A tuple containing: (Distance Matrix, Object List, Dendrogram, Medoids)

    EXAMPLES
        calc_univariate_hca group_name.D.*, linkage_method=ward
        calc_univariate_hca *.CS.*
    
    SEE ALSO
        calc_mutivariate_hca
    """

    objects = pm.get_object_list(sele)
    assert objects is not None and len(objects) >= 2, "At least two hotspots are required for comparison."

    X = []
    obj_coords = {}
    for obj in objects:
        if obj not in obj_coords:
            obj_coords[obj] = get_coords(obj)
    
    for idx1, obj1 in enumerate(objects):
        for idx2, obj2 in enumerate(objects):
            if idx1 >= idx2:
                continue
            coords1 = obj_coords[obj1]
            coords2 = obj_coords[obj2]
            match overlap_function:
                case HcaOverlapFunction.FO_AVG:
                    fo1 = get_fo(coords1, coords2, radius=radius)
                    fo2 = get_fo(coords2, coords1, radius=radius)
                    ret = (fo1 + fo2) / 2
                case HcaOverlapFunction.JACCARD:
                    # ret = res_sim(
                    #     obj1,
                    #     obj2,
                    #     radius=radius,
                    #     method=ResidueSimilarityMethod.JACCARD,
                    #     seq_align=seq_align_before_overlap,
                    # )
                    raise NotImplementedError("JACCARD similarity is not yet implemented.")
                    pass
                case HcaOverlapFunction.OVERLAP:
                    # ret = res_sim(
                    #     obj1,
                    #     obj2,
                    #     radius=radius,
                    #     method=ResidueSimilarityMethod.OVERLAP,
                    #     seq_align=seq_align_before_overlap,
                    # )
                    raise NotImplementedError("OVERLAP similarity is not yet implemented.")
                    pass
            X.append(1 - ret)
    dendro, medoids = plot_hca_base(
        X, objects, linkage_method,
        nclusters=nclusters,
        color_threshold=color_threshold,
        only_medoids=only_medoids,
        annotate=annotate,
        vmin=0,
        vmax=1,
        rename_leafs=rename_leafs,
        figure_title=figure_title,
        dendrogram_plot=dendrogram_plot,
        heatmap_plot=heatmap_plot,
    )
    return X, objects, dendro, medoids



class OverlapFunction(StrEnum):
    FO = "fo"
    FO_AVG = "fo_avg"
    DC = "dc"
    DCE = "dce"


@new_command
def calc_overlap_matrix(
    sele_a: str,
    sele_b: Optional[str] = None,
    function: OverlapFunction = OverlapFunction.FO,
    radius: float = 2.0,
    annotate: bool = False,
    rename_leafs: Optional[Dict[str, str]] = None,
    linkage_method: Optional[LinkageMethod] = None,
):
    """
    DESCRIPTION

        Generates a heatmap matrix visualizing the overlap or contact 
        metrics between two groups of PyMOL objects. 

        This is ideal for cross-comparing FTMap hotspots across different 
        protein conformations or comparing a ligand to a set of probe clusters.

    USAGE

        plot_overlap_matrix sele_a [, sele_b [, function [, radius [, annotate]]]]

    ARGUMENTS

        sele_a: str
            The selection for the vertical axis (rows).

        sele_b: str, optional
            The selection for the horizontal axis (columns). If omitted or 
            blank, it defaults to sele_a (creating a self-comparison matrix).

        function: OverlapFunction, default=OverlapFunction.FO
            The metric to calculate. Supported values:
            - 'FO': Fraction of Overlap [0.0 - 1.0]
            - 'DC': Distance Contacts (raw count)
            - 'DCE': Distance Contact Efficiency (normalized)

        radius: float, default=2.0
            The distance cutoff (Angstroms) passed to the overlap function.

        annotate: bool, default=False
            If True, writes the numerical values directly inside the heatmap 
            cells. Text color (black/white) is automatically adjusted for 
            legibility based on the cell intensity.
        
        rename_leafs:
            A dictionary mapping PyMOL object names to user-friendly labels.
        
        linkage_method:
            Optional clustering algorithm.

    RETURNS

        A pandas DataFrame containing the raw data with columns ['A', 'B', 'METRIC'].

    NOTES
        If function is 'FO', the color scale is fixed between 0.0 and 1.0.

    EXAMPLE

        # Compare hotspots in group 1 vs group 2
        plot_overlap_matrix group1.D*, group1.B*, function=FO, annotate=True

        # Create a self-similarity matrix for all objects in a session
        plot_overlap_matrix *, function=DC
    """
    objs_a = pm.get_object_list(sele_a)
    if isinstance(sele_b, str) and sele_b.strip():
        objs_b = pm.get_object_list(sele_b) or []
    else:
        objs_b = objs_a or []
    
    match function:
        case OverlapFunction.FO:
            get_value = get_fo
        case OverlapFunction.FO_AVG:
            get_value = lambda a, b, radius=radius: (get_fo(a, b, radius)+get_fo(b, a, radius))/2
        case OverlapFunction.DC:
            get_value = get_dc
        case OverlapFunction.DCE:
            get_value = get_dce

    ret = []
    X = []
    
    obj_coords = {}
    for obj in [*objs_a, *objs_b]:
        if obj not in obj_coords:
            obj_coords[obj] = get_coords(obj)
    
    for i1, a in enumerate(objs_a):
        row = []
        for i2, b in enumerate(objs_b):
            value = get_value(obj_coords[a], obj_coords[b], radius=radius)
            row.append(value)
            ret.append([a, b, value])
        X.append(row)
    
    X = np.array(X)
    if linkage_method and len(X) > 2:
        Z_rows = linkage(X, method=linkage_method)
        idx_rows = leaves_list(Z_rows)
    else:
        idx_rows = np.arange(len(X))

    if linkage_method and len(X.T) > 2:
        Z_cols = linkage(X.T, method=linkage_method)
        idx_cols = leaves_list(Z_cols)
    else:
        idx_cols = np.arange(len(X.T))
    
    X = X[idx_rows, :][:, idx_cols]

    fig, ax = plt.subplots(constrained_layout=True)
    
    objs_a_lbl = []
    for obj_a in objs_a:
        new_lbl = (rename_leafs or {}).get(obj_a, obj_a)
        objs_a_lbl.append(new_lbl)
    
    objs_b_lbl = []
    for obj_b in objs_b:
        new_lbl = (rename_leafs or {}).get(obj_b, obj_b)
        objs_b_lbl.append(new_lbl)
        
    ax.set_yticks(range(len(objs_a)), np.array(objs_a_lbl)[idx_rows])
    ax.set_xticks(range(len(objs_b)), np.array(objs_b_lbl)[idx_cols])
    
    ax.tick_params(axis="x", rotation=90)
    if function in [OverlapFunction.FO, OverlapFunction.FO_AVG]:
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
        for i1 in range(len(objs_a)):
            for i2 in range(len(objs_b)):
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


def calc_ligand_fit(
    hs_sele: Selection,
    ligs_sele: Selection,
    function: OverlapFunction,
    radius: float,
    annotate: bool,
    lig_metric: BindMetric,
    bind_df: Any,
):
    if len(pm.get_object_list(hs_sele)) != 1:
        raise ValueError("Only one hotspot can be analyzed at time.")
    
    
    objs_hss = pm.get_object_list(hs_sele)
    objs_ligs = pm.get_object_list(ligs_sele)
    
    match function:
        case OverlapFunction.FO:
            get_value = get_fo
        case OverlapFunction.FO_AVG:
            get_value = lambda a, b, radius=radius: (get_fo(a, b, radius)+get_fo(b, a, radius))/2
        case OverlapFunction.DC:
            get_value = get_dc
        case OverlapFunction.DCE:
            get_value = get_dce
    ret = []
    X = []
    obj_coords = {}
    for obj in [*objs_hss, *objs_ligs]:
        if obj not in obj_coords:
            obj_coords[obj] = get_coords(obj)
    
    for i1, a in enumerate(objs_hss):
        row = []
        for i2, b in enumerate(objs_ligs):
            value = get_value(obj_coords[a], obj_coords[b], radius=radius)
            row.append(value)
            ret.append([a, b, value])
        X.append(row)
    
    overlap_df = pd.DataFrame.from_records(ret, columns=['A', 'B', function.upper()])
    overlap_df = overlap_df.rename(columns={'B': 'Ligand'})
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
    if annotate:
        rows = zip(x, y, df['Ligand'], df['Label'])
        for x, y, obj, label in rows:
            s = label.strip() or obj
            ax.text(x, y, s)
    ax.set_xlabel(f"{function_col} / {function_col}_ref")
    ax.set_ylabel(f"{lig_metric} / {lig_metric}_ref")


@new_command
def calc_fingerprints(
    multi_seles: str,
    site: Selection = "*",
    site_radius: float = 5.0,
    omega_conservation: str = "*:.",
    contact_radius: float = 4.0,
    nbins: int = 5,
    sharex: bool = True,
    linkage_method: LinkageMethod = LinkageMethod.WARD,
    color_threshold: float = -1.0,
    nclusters: int = -1,
    only_medoids: bool = False,
    annotate: bool = True,
    share_ylim: bool = True,
    figure_title: str | None = None,
    fingerprints_plot: str | Path | axes.Axes | None = None,
    dendrogram_plot: str | Path | axes.Axes | None = None,
    heatmap_plot: str | Path | axes.Axes | None = None,
    quiet: bool = True,
):
    """
    DESCRIPTION

        Computes the residue contact fingerprint for multiple hotspot or
        consensus sites selections, maps them via sequence alignment, and
        performs a Hierarchical Cluster Analysis (HCA) based on Pearson
        correlation distances.

        The method aligns the underlying protein polymers using Clustal Omega,
        extracts structural atom contacts within a defined radius around the 
        target interaction site.

    ARGUMENTS

        multi_seles: str
            A slash-separated string of PyMOL selections containing the objects 
            to compare (e.g., 'hs_or_cs_1 / hs_or_cs_2'). They must came from
            load_ftmap and belongs to a protein group.

        site: Selection, default="*"
            A PyMOL selection used to focus the fingerprint sub-region based on 
            the first protein structure.

        site_radius: float, default=5.0
            Distance cutoff (Angstroms) to include residues in fingerprint
            relative to the 'site' selection.

        omega_conservation: str, default="*:."
            Clustal Omega conservation string match criteria for filtering residues.

        contact_radius: float, default=4.0
            Distance cutoff (Angstroms) used to compute raw atomic contacts 
            between the hotspot/cs and target residues.

        nbins: int, default=5
            Number of bins/labels applied to the x-axis tick locator.

        sharex: bool, default=True
            If True, subplots share the same x-axis layout, hiding inner labels 
            to prevent visual clutter.

        linkage_method: LinkageMethod, default='ward'
            The clustering linkage algorithm used to construct the dendrogram ('ward',
            'single', 'complete').

        color_threshold: float, default=-1.0
            Distance cutoff for coloring dendrogram branches. Disabled if negative.
            Can be used only if nclusters is disabled.

        nclusters: int, default=-1
            Target number of clusters to coloring dendrogram branches. Disable if zero
            or less. Can be used only if color_threshold is disabled.

        only_medoids: bool, default=False
            If True, restricts the final HCA visualization strictly to cluster medoids.

        annotate: bool, default=True
            If True, writes numerical values inside the distance matrix heatmap cells.

        share_ylim: bool, default=True
            If True, synchronizes the y-axis maximum scale across all fingerprint 
            bar charts for direct visual comparison.

        figure_title: str, optional
            Title text displayed at the top of the generated figure window.

        fingerprints_plot: str, Path, Axes, optional
            Target destination for the bar charts. Can be a Matplotlib Axes, 
            a file path to export the image, or a boolean.

        dendrogram_plot: str, Path, Axes, optional
            Target destination for the HCA dendrogram plot layout.

        heatmap_plot: str, Path, Axes, optional
            Target destination for the Pearson correlation distance matrix heatmap.

        quiet: bool, default=True
            If True, suppresses console verbosity and raw stdout outputs.

    RETURNS

        A tuple containing: (Fingerprints List, Correlation Matrix, Dendrogram, Medoids)

    EXAMPLES

        # Compare specific hotspots across two structures separated by a slash
        calc_fingerprints 8DSU.K15_D_01* / 6XHM.K15_D_01*, linkage_method=ward

        # Focus fingerprints on a specific binding site pocket with custom binning
        calc_fingerprints 8DSU.CS_* / 6XHM.CS_*, site=resi 8-101, nbins=10

    SEE ALSO

        calc_univariate_hca, calc_mutivariate_hca
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
    
    ref_sele = seles[0]
    ref_polymer = polymers[0]
    site_sele = f"{ref_polymer} & ({ref_polymer} within {site_radius} of ({site}))"
    site_resis = []
    for at in pm.get_model(f"({site_sele}) & present & guide & polymer").atom:
        site_resis.append((at.model, at.index))
    
    mapping = clustal_omega(
        polymers,
        omega_conservation.strip(),
        titles=seles
    )

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

    if fingerprints_plot:
        if isinstance(fingerprints_plot, (str, Path)) or fingerprints_plot is True:
            _, fpt_axs = plt.subplots(nrows=len(seles))

        elif isinstance(fingerprints_plot, axes.Axes):
            fpt_axs = []
            height = 1/len(fpt)
            for i, _ in enumerate(sele):
                ax = fingerprints_plot.inset_axes([0, (i+1)*height], 1, height)
                fpt_axs.append(ax)
    else:
        fpt_axs = None
    
    if not isinstance(fpt_axs, (np.ndarray, list)):
        fpt_axs = [fpt_axs]
        
    if not all([len(fpts[0]) == len(fpt) for fpt in fpts]):
        raise ValueError(
            "All fingerprints must have the same length. "
            "Do you have incomplete structures?"
        )
    
    max_val = 0
    for ix, (ax, fpt, sele) in enumerate(zip(fpt_axs, fpts, seles)):
        labels = ["%s%s %s_%s" % k for k in fpt]
        if sharex and ix == 0:
            shared_labels = labels
        elif sharex and ix + 1 == len(seles):
            labels = shared_labels
        arange = np.arange(len(fpt))
        max_val = max(max(fpt.values()) if fpt else 0, max_val)
        ax.bar(arange, fpt.values())
        ax.set_title(sele)
        ax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
        if sharex and ix + 1 < len(seles):
            ax.set_xticks([])
        if not sharex or ix + 1 == len(seles):
            ax.set_xticks(arange, labels=labels, rotation=90)
            ax.locator_params(axis="x", tight=True, nbins=nbins)
            for label in ax.xaxis.get_majorticklabels():
                label.set_verticalalignment("top")
    if share_ylim:
        for ax in fpt_axs:
            ax.set_ylim(0, max_val * 1.05)
    
    if fingerprints_plot:
        fig = fpt_axs[0].get_figure(True)
        fig.set_layout_engine('compressed')
        fig.supylabel('Atom Counts')
        if figure_title:
            fig.suptitle(figure_title)
        if isinstance(fingerprints_plot, (str, Path)):
            fig.savefig(str(fingerprints_plot))
        elif fingerprints_plot is True:
            fig.show()
        elif isinstance(fingerprints_plot, axes.Axes):
            pass

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

    dendro, medoids = plot_hca_base(
        corrs,
        labels,
        linkage_method=linkage_method,
        color_threshold=color_threshold,
        nclusters=nclusters,
        only_medoids=only_medoids,
        annotate=annotate,
        vmin=0,
        vmax=2,
        figure_title=figure_title,
        dendrogram_plot=dendrogram_plot,
        heatmap_plot=heatmap_plot
    )
    return fpts, corrs, dendro, medoids


class ResidueSimilarityMethod(StrEnum):
    JACCARD = "jaccard"
    OVERLAP = "overlap"


@new_command
def res_sim(
    hs1: Selection,
    hs2: Selection,
    radius: float = 4.0,
    seq_align: bool = False,
    align_method: AligMethod = AligMethod.CEALIGN,
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
    group1 = hs1.rsplit(".", maxsplit=2)[0]
    group2 = hs2.rsplit(".", maxsplit=2)[0]

    sel1 = f"{group1}.protein within {radius} from ({hs1})"
    sel2 = f"{group2}.protein within {radius} from ({hs2})"

    resis1 = set()
    for at in pm.get_model(sel1).atom:
        resis1.add((at.chain, at.resi))

    if group1 == group2 or not seq_align:
        pymol.stored.resis2 = set()
        pm.iterate_state(1, sel2, "stored.resis2.add((chain, resi))")
        resis2 = pymol.stored.resis2
    else:
        try:
            # FIXME Clustal Omega?
            aln_obj = pm.get_unused_name()
            pm.extra_fit(
                f"{group1}.protein",
                f"{group2}.protein",
                method=str(align_method),
                transform=0,
                object=aln_obj
            )
            raw = pm.get_raw_alignment(aln_obj)

            resis = {}
            pm.iterate_state(
                1, aln_obj, "resis[model, index] = (chain, resi)", space={"resis": resis}
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

    pm.alter(protein, "p.cnt_atoms=0")
    for prot_atom in pm.get_model(f"({protein}) within {radius} of ({sel})").atom:
        match type:
            case PrioritizationType.RESIDUE:
                prot_atom_sel = f"byres index {prot_atom.index}"
            case PrioritizationType.ATOM:
                prot_atom_sel = f"index {prot_atom.index}"
        cnt = pm.count_atoms(f"({sel}) within {radius} of ({prot_atom_sel})")
        pm.alter(prot_atom_sel, f"p.cnt_atoms={cnt}")

    pm.hide("everything", protein)
    pm.show("cartoon", protein)
    match type:
        case PrioritizationType.RESIDUE:
            # pm.show("cartoon", protein)
            # pm.show("surface", protein)
            pass
        case PrioritizationType.ATOM:
            pm.show("sticks", "p.cnt_atoms>0")
    pm.spectrum("p.cnt_atoms", palette=palette, selection=protein)


class DistanceMethod(StrEnum):
    EUCLIDEAN = "euclidean"
    CITYBLOCK = "cityblock"
    DICE = "dice"


@new_command
def calc_multivariate_hca(
    sele: Selection,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: float = -1.0,
    nclusters: int = -1.0,
    only_medoids: bool = False,
    annotate: bool = False,
    dist_method: DistanceMethod = DistanceMethod.EUCLIDEAN,
    rename_leafs: Optional[Dict[str, str]] = None,
    figure_title: str | None = None,
    dendrogram_plot: str | Path | bool | axes.Axes | None = None,
    heatmap_plot: str | Path | bool | axes.Axes | None = None,
):
    """
    DESCRIPTION

        Performs a Multivariate Hierarchical Cluster Analysis (HCA) on FTMap
        consensus sites or hotspots.

        The command automatically extracts properties stored within PyMOL objects
        and center-of-mass coordinates, then constructs standardized feature vectors
        (Z-score) for each object to calculate high-dimensional distances and
        a dendogram and heatmap.

    PROPERTIES EVALUATED

        Feature vectors are built based on the object type:
        - Hotspots: ST, S0, CD, MD and XYZ (7 variables)
        - Consensus sites: S + XYZ Coordinates (4 variables)
        - ACS (Atomic Contact Surfaces): ST, MD + XYZ Coordinates (5 variables)

    ARGUMENTS

        sele:
            A PyMOL selection containing the objects to compare. All objects must
            be of the same type (e.g., all hotspots or all consensus sites).

        linkage_method:
            The clustering linkage algorithm used to compute the dendrogram 
            (e.g., SINGLE, COMPLETE, WARD).

        color_threshold:
            Distance cutoff for coloring dendrogram branches.

        nclusters:
            Target number of clusters to set the color threshold. Use one or other.

        only_medoids:
            If True, focuses the analysis or visualization only on the cluster medoids.

        annotate:
            If True, adds value annotations or labels to the heatmap cells.

        dist_method:
            The distance metric used to calculate the distance matrix (e.g., EUCLIDEAN).

        rename_leafs:
            A dictionary mapping internal PyMOL object names to user-friendly labels.

        figure_title:
            Title text displayed at the top of the generated figure window.

        dendrogram_plot:
            Target destination for the dendrogram. Can be a Matplotlib Axes object, 
            a file path (str/Path) to save the plot, a boolean, or None.

        heatmap_plot:
            Target destination for the distance matrix heatmap. Can be a Matplotlib 
            Axes object, a file path (str/Path) to save the plot, a boolean, or None.

    RETURNS

        A tuple containing: (Distance Matrix, Object List, Dendrogram, Medoids)

    EXAMPLES

        calc_multivariate_hca group_name.D.*, linkage_method=ward
        calc_multivariate_hca *.CS.*
    """
    object_list = pm.get_object_list(sele)
    assert object_list is not None and len(object_list) >= 2, "At least two hotspots are required for comparison."
    assert len(set(pm.get_property("Type", o) for o in object_list)) == 1, "Only hotspots or only consensus sites are supported in the HCA."

    hs_type = pm.get_property("Type", object_list[0])
    if hs_type == "HS":
        n_props = 4
    elif hs_type == "CS":
        n_props = 1
    labels = []

    p = np.zeros((len(object_list), n_props + 3))

    for ix, obj in enumerate(object_list):
        labels.append(obj)
        x, y, z = pm.get_coordset(obj).mean(axis=0) # FIXME use pm.centerofmass instead of mean
        if hs_type == "HS":
            ST = pm.get_property("ST", obj)
            S0 = pm.get_property("S0", obj)
            CD = pm.get_property("CD", obj)
            MD = pm.get_property("MD", obj)
            p[ix, :] = np.array([ST, S0, CD, MD, x, y, z])
        elif hs_type == "CS":
            S = pm.get_property("S", obj)
            p[ix, :] = np.array([S, x, y, z])
        elif hs_type == "ACS":
            ST = pm.get_property("ST", obj)
            MD = pm.get_property("MD", obj)
            p[ix, :] = np.array([ST, MD, x, y, z])
    
    p = (p - p.mean(axis=0)) / (p.std(axis=0) + 1e-8)
    X = distance.pdist(p, dist_method)
    dendro, medoids = plot_hca_base(
        X,
        labels=labels,
        linkage_method=linkage_method,
        color_threshold=color_threshold,
        nclusters=nclusters,
        only_medoids=only_medoids,
        annotate=annotate,
        rename_leafs=rename_leafs or {},
        figure_title=figure_title,
        heatmap_plot=heatmap_plot,
        dendrogram_plot=dendrogram_plot,
    )
    return X, object_list, dendro, medoids


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
    overlap_df = calc_overlap_matrix(
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

        self.deepSearch = QCheckBox()
        self.deepSearch.setChecked(False)
        boxLayout.addRow("Deep search:", self.deepSearch)

        self.removeNested = QCheckBox()
        self.removeNested.setChecked(False)
        boxLayout.addRow("Remove nested:", self.removeNested)

        self.maxCollisions = QDoubleSpinBox()
        self.maxCollisions.setRange(0.0, 1.0)
        self.maxCollisions.setSingleStep(0.05)
        self.maxCollisions.setValue(0.10)
        boxLayout.addRow("Max collisions:", self.maxCollisions)
        
        self.pretty = QCheckBox()
        self.pretty.setChecked(False)
        boxLayout.addRow("Pretty session:", self.pretty)
        
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
        deep_search = self.deepSearch.isChecked()
        remove_nested = self.removeNested.isChecked()
        max_collisions = self.maxCollisions.value()
        pretty = self.pretty.isChecked()
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
        
        for filename, group in zip(filenames, groups):
            load_ftmap(
                filename=filename,
                group=group,
                deep_search=deep_search,
                remove_nested=remove_nested,
                clash_threshold=max_collisions,
                pretty=pretty
            )


class SortableItem(QTableWidgetItem):
    def __init__(self, obj, stringfy):
        super().__init__()
        self.setData(QtCore.Qt.ItemDataRole.EditRole, obj)
        self.setData(QtCore.Qt.ItemDataRole.DisplayRole, stringfy(obj))
            
    def __lt__(self, other):
        this = self.data(QtCore.Qt.ItemDataRole.EditRole)
        that = other.data(QtCore.Qt.ItemDataRole.EditRole)
        return this < that


class OptionalPositiveDoubleDelegate(QStyledItemDelegate):
    """For the winner of ugliest API design championshop ever."""
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
                "S1",
                "SZ",
                "CD",
                "MD",
                "Length",
                "Kavity",
            ],
            ("CS", "CS"): ["S"],
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
                        if isinstance(obj, float):
                            stringfy = lambda o: f'{o:.2f}'
                        else:
                            stringfy = str
                        self.appendRow(title, key, obj, stringfy)

            self.tables[title].setSortingEnabled(True)

    def appendRow(self, title, key, obj, stringfy):
        self.tables[title].insertRow(self.tables[title].rowCount())
        line = self.tables[title].rowCount() - 1

        self.tables[title].setItem(line, 0, SortableItem(obj, stringfy))

        for idx, prop in enumerate(self.hotspotsMap[(title, key)]):
            prop_value = pm.get_property(prop, obj)
            self.tables[title].setItem(line, idx + 1, SortableItem(prop_value, stringfy))

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

        self.hotspotSeleLine = QLineEdit("")
        self.hotspotSeleLine.setPlaceholderText("PyMOL Selection Algebra")
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
        
        groupBox = QGroupBox("Univariate analysis")
        
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.univariateFunctionCombo = QComboBox()
        self.univariateFunctionCombo.addItems([e.value for e in HcaOverlapFunction])
        boxLayout.addRow("Function:", self.univariateFunctionCombo)

        self.radiusSpin = QDoubleSpinBox()
        self.radiusSpin.setValue(2)
        self.radiusSpin.setSingleStep(0.5)
        self.radiusSpin.setDecimals(2)
        self.radiusSpin.setMinimum(1)
        self.radiusSpin.setMaximum(10)
        boxLayout.addRow("Radius:", self.radiusSpin)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_univariate_hca)
        boxLayout.addWidget(plotButton)

        groupBox = QGroupBox("Multivariate analysis")
        
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)
        self.multivariateFunctionCombo = QComboBox()
        self.multivariateFunctionCombo.addItems([e.value for e in DistanceMethod])
        boxLayout.addRow("Distance function:", self.multivariateFunctionCombo)

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
        heatmap_plot = self.enableHeatmapCheck.isChecked()

        calc_multivariate_hca(
            sele=sele,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            only_medoids=only_medoids,
            annotate=annotate,
            rename_leafs=self.getLeafLabels(),
            heatmap_plot=heatmap_plot
        )
        plt.show()
    
    def plot_univariate_hca(self):
        sele = self.hotspotSeleLine.text()
        overlap_function = self.univariateFunctionCombo.currentText()
        radius = self.radiusSpin.value()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        only_medoids = self.onlyMedoidsCheck.isChecked()
        annotate = self.annotateCheck.isChecked()

        calc_univariate_hca(
            sele=sele,
            radius=radius,
            overlap_function=overlap_function,
            annotate=annotate,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            only_medoids=only_medoids,
            rename_leafs=self.getLeafLabels(),
        )
        plt.show()


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
            obj = self.item(r, 0).text().strip()
            pki = self.item(r, 1).text().strip()
            ha = self.item(r, 5).text().strip()
            mw = self.item(r, 6).text().strip()
            label = self.item(r, 7).text().strip()
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
                le = self.item(r, 2).text().strip()
                bei = self.item(r, 3).text().strip()
                fq = self.item(r, 4).text().strip()
                
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
        self.ligMetricCombo.addItems([e.value for e in BindMetric])
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

        calc_ligand_fit(
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

        self.linkageMethodCombo = QComboBox()
        self.linkageMethodCombo.addItems(['none'] + [e.value for e in LinkageMethod])
        layout.addRow("Linkage method:", self.linkageMethodCombo)

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
        linkage_method = self.linkageMethodCombo.currentText()
        if linkage_method.lower() == 'none':
            linkage_method = None

        calc_overlap_matrix(
            sele_a=sele_a,
            sele_b=sele_b,
            function=function,
            radius=radius,
            annotate=annotate,
            linkage_method=linkage_method
        )
        plt.show()
    
    def export_overlap(self):
        sele_a = self.aSeleLine.text().strip()
        sele_b = self.bSeleLine.text().strip()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        annotate = self.annotateCheck.isChecked()
        
        table_df = calc_overlap_matrix(
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



class FingerprintWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        self.setLayout(layout)

        groupBox = QGroupBox("Visualization")
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
        fingerprints_plot = self.fingerprintsCheck.isChecked()
        dendrogram_plot = self.hcaCheck.isChecked()
        omega_conservation = self.omegaConservation.text().strip()
        nbins = self.nBinsSpin.value()
        share_ylim = self.shareYLimCheck.isChecked()
        sharex = self.sharexCheck.isChecked()
        annotate = self.annotateCheck.isChecked()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        only_medoids = self.onlyMedoidsCheck.isChecked()
        calc_fingerprints(
            multi_seles,
            site,
            site_radius,
            contact_radius=contact_radius,
            omega_conservation=omega_conservation,
            nbins=nbins,
            sharex=sharex,
            share_ylim=share_ylim,
            linkage_method=linkage_method,
            only_medoids=only_medoids,
            annotate=annotate,
            fingerprints_plot=fingerprints_plot,
            # heatmap_plot=heatmap_plot,
            dendrogram_plot=dendrogram_plot,
            color_threshold=color_threshold,
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
        tab.addTab(HcaWidget(), "Hotspot Similarity")
        tab.addTab(OverlapWidget(), "Overlap Matrix")
        tab.addTab(LigandFitWidget(), "Ligand Fit")
        tab.addTab(FingerprintWidget(), "Fingerprints")

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
