import os.path
import shutil
import subprocess
import re
import tempfile
from glob import glob
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.spatial import distance_matrix, distance
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from strenum import StrEnum

from .utils import (
    ONE_LETTER, dendrogram, declare_command, Selection, multiple_expression_selector,
    mpl_axis, expression_selector, plot_hca_base
)
from .mapping import get_mapping

from pymol import cmd as pm

matplotlib.use("Qt5Agg")


def get_clusters():
    clusters = []
    eclusters = []

    for obj in pm.get_object_list():
        if obj.startswith(f"crosscluster."):
            _, _, s, _ = obj.split(".", maxsplit=4)
            coords = pm.get_coords(obj)
            clusters.append(
                SimpleNamespace(
                    selection=obj,
                    strength=int(s),
                    coords=coords,
                )
            )
        elif obj.startswith("consensus."):
            _, _, s = obj.split(".", maxsplit=3)
            coords = pm.get_coords(obj)
            clusters.append(
                SimpleNamespace(
                    selection=obj,
                    strength=int(s),
                    coords=coords,
                )
            )

        elif obj.startswith("clust."):
            _, idx, s, probe_type = obj.split(".", maxsplit=4)
            eclusters.append(
                SimpleNamespace(
                    selection=obj,
                    probe_type=probe_type,
                    strength=int(s),
                    idx=int(idx),
                )
            )
    return clusters, eclusters


def set_properties(obj, obj_name, properties):
    for prop, value in properties.items():
        pm.set_property(prop, value, obj_name)
        pm.set_atom_property(prop, value, obj_name)
        setattr(obj, prop, value)


def get_kozakov2015(group, clusters, max_length):
    k15 = []
    for length in range(max_length, 1, -1):
        for combination in combinations(clusters, length):
            cd = []
            cluster1 = combination[0]
            avg1 = np.average(cluster1.coords, axis=0)
            for cluster2 in combination:
                avg2 = np.average(cluster2.coords, axis=0)
                cd.append(distance.euclidean(avg1, avg2))

            coords = np.concatenate([c.coords for c in combination])
            max_coord = coords.max(axis=0)
            min_coord = coords.min(axis=0)

            selection = " or ".join(c.selection for c in combination)
            hs = SimpleNamespace(
                selection=selection,
                clusters=combination,
                kozakov_class=None,
                strength=sum(c.strength for c in combination),
                strength0=combination[0].strength,
                center_center=np.max(cd),
                max_dist=distance.euclidean(max_coord, min_coord),
                length=len(combination),
            )
            s0 = hs.clusters[0].strength
            sz = hs.clusters[-1].strength
            cd = hs.center_center
            md = hs.max_dist

            if s0 < 13 or md < 7 or sz <= 5:
                continue
            if s0 >= 16 and cd < 8 and md >= 10:
                hs.kozakov_class = "D"
            if s0 >= 16 and cd < 8 and 7 <= md < 10:
                hs.kozakov_class = "DS"
            if 13 <= s0 < 16 and cd < 8 and md >= 10:
                hs.kozakov_class = "B"
            if 13 <= s0 < 16 and cd < 8 and 7 <= md < 10:
                hs.kozakov_class = "BS"

            if hs.kozakov_class:
                k15.append(hs)

    k15 = sorted(k15, key=lambda hs: (-hs.strength0, -hs.strength))
    k15 = sorted(k15, key=lambda hs: ["D", "DS", "B", "BS"].index(hs.kozakov_class))
    k15 = list(k15)

    for idx, hs in enumerate(k15):
        new_name = f"{group}.K15_{hs.kozakov_class}_{idx:02}"
        pm.create(new_name, hs.selection)
        pm.group(group, new_name)
        hs.selection = new_name
        set_properties(hs, new_name, {
            "Type": "K15",
            "Group": group,
            'Selection': new_name,
            "Class": hs.kozakov_class,
            "S": hs.strength,
            "S0": hs.strength0,
            "CD": round(hs.center_center, 2),
            "MD": round(hs.max_dist, 2),
            "Length": hs.length,
        })
    return k15

def get_fpocket(group, protein):
    pockets = []
    # FIXME TODO
    with tempfile.TemporaryDirectory() as tempdir:
        protein_pdb = f"{tempdir}/{group}.pdb"
        pm.save(protein_pdb, selection=protein)
        subprocess.check_call(
            [
                shutil.which('fpocket'),
                '-f',
                protein_pdb
            ],
        )
        header_re = re.compile(r'^HEADER\s+\d+\s+-(.*):(.*)$')
        for pocket_pdb in glob(f"{tempdir}/{group}_out/pockets/pocket*_atm.pdb"):
            idx = os.path.basename(pocket_pdb).replace('pocket', '').replace('_atm.pdb', '')
            idx = int(idx)
            pocket = SimpleNamespace(
                selection = f'{group}.fpocket_{idx:02}'
            )
            pm.delete(pocket.selection)
            pm.load(pocket_pdb, pocket.selection)
            pm.set_property("Type", "Fpocket", pocket.selection)
            pm.set_property("Group", group. pocket.selection)
            pm.set_property("Selection", pocket.select)
            for line in pm.get_property('pdb_header', pocket.selection).split('\n'):
                if match := header_re.match(line):
                    prop = match.group(1).strip()
                    value = float(match.group(2))
                    setattr(pocket, prop, value)
                    pm.set_property(prop, value, pocket.selection)
            pockets.append(pocket)

    return pockets



def process_clusters(group, clusters):
    for idx, cs in enumerate(clusters):
        new_name = f"{group}.CS_{idx:02}"
        pm.create(new_name, cs.selection)
        cs.selection = new_name
        pm.group(group, new_name)
        set_properties(cs, new_name, {
            "Type": "CS",
            "Group": group,
            "Selection": new_name,
            "S": cs.strength,
        })
    pm.delete("consensus.*")
    pm.delete("crosscluster.*")


def process_eclusters(group, eclusters):
    for acs in eclusters:
        new_name = f"{group}.ACS_{acs.probe_type}_{acs.idx:02}"
        pm.create(new_name, acs.selection)
        acs.selection = new_name
        pm.group(group, new_name)

        coords = pm.get_coords(new_name)
        md = distance_matrix(coords, coords).max()

        set_properties(acs, new_name, {
            "Type": "ACS", 
            "Group": group, 
            "Selection": new_name,
            "Class": acs.probe_type, 
            "S": acs.strength, 
            "MD": round(md, 2)
        })
    pm.delete("clust.*")


def get_egbert2019(group, fpo_list, clusters):
    e19 = []
    idx_e19 = 0
    for pocket in fpo_list:
        sel = f"byobject ({group}.CS_* within 4 of {pocket.selection})"
        objs = pm.get_object_list(sel)
        if len(objs) > 3 and sum([pm.get_property("S", o) >= 16 for o in objs]) > 2:
            new_name = f"{group}.C_{idx_e19:02}"
            pm.create(new_name, sel)
            pm.group(group, new_name)

            s_list = [pm.get_property("S", o) for o in objs]
            pm.set_property("Type", "Egbert2019", new_name)
            pm.set_property("Group", group, new_name)
            pm.set_property("Selection", sel, new_name)
            pm.set_property("Fpocket", pocket.selection, new_name)
            pm.set_property("S", sum(s_list), new_name)
            pm.set_property("S0", s_list[0])
            pm.set_property("S1", s_list[1])
            pm.set_property("Length", len(objs), new_name)
            e19.append(SimpleNamespace(selection=new_name))
            idx_e19 += 1
    return e19

def get_kozakov2015_large(group, fpo_list, clusters):
    k15l = []
    idx = 0
    for pocket in fpo_list:
        sel = f"byobject ({group}.CS_* within 4 of {pocket.selection})"
        objs = pm.get_object_list(sel)
        if len(objs) == 0:
            continue
        pocket_clusters = [c for c in clusters if c.selection in objs]
        i_s0 = np.argmax(c.strength for c in pocket_clusters)
        s0 = pocket_clusters[i_s0].strength

        cd = []
        cluster1 = pocket_clusters[i_s0]
        avg1 = np.average(cluster1.coords, axis=0)
        for cluster2 in pocket_clusters:
            avg2 = np.average(cluster2.coords, axis=0)
            cd.append(distance.euclidean(avg1, avg2))
        cd = max(cd)

        coords = np.concatenate([c.coords for c in pocket_clusters])
        max_coord = coords.max(axis=0)
        min_coord = coords.min(axis=0)
        md = distance.euclidean(max_coord, min_coord)

        strength = sum(c.strength for c in pocket_clusters)

        i_sz = np.argmin(c.strength for c in pocket_clusters)
        sz = pocket_clusters[i_sz].strength

        if 13 <= s0 < 16 and cd >=8 and md >= 10 and sz >= 5:
            new_name = f"{group}.K15_BL_{idx:02}"
            klass = "BL"
        if s0 >= 16 and cd >=8 and md >= 10 and sz >= 5:
            new_name = f"{group}.K15_DL_{idx:02}"
            klass = "DL"
        else:
            continue
        idx += 1

        hs = SimpleNamespace()
        hs.kozakov_class = klass
        pm.create(new_name, sel)
        pm.group(group, new_name)
        set_properties(hs, new_name, {
            "Type": "K15",
            "Group": group,
            'Selection': new_name,
            "Class": klass,
            "S": strength,
            "S0": s0,
            "CD": round(cd, 2),
            "MD": round(md, 2),
            "Length": len(pocket_clusters),
            "Fpocket": pocket.selection
        })

        if hs.kozakov_class:
            k15l.append(hs)
        
    return k15l


@declare_command
def load_ftmap(
    filename: Path,
    group: str = "",
    k15_max_length: int = 5,
    run_fpocket: bool = False
):
    """
    Load a FTMap PDB file and classify hotspot ensembles in accordance to
    Kozakov et al. (2015).
    https://doi.org/10.1021/acs.jmedchem.5b00586

    OPTIONS
        filename        mapping PDB file.
        group           optional group name.
        k15_max_length  max hotspot length for Kozakov15 hotspots (default: 8).

    EXAMPLES
        load_ftmap ace_example.pdb
        load_ftmap ace_example.pdb, group=MyProtein
        load_ftmap 3PO1.pdb, k15_max_length=5
    """
    if not group:
        group = os.path.splitext(os.path.basename(filename))[0]
    group = group.replace(".", "_")

    pm.delete(f"%{group}")
    pm.load(filename, quiet=1)

    if objs := pm.get_object_list("*_protein"):
        assert len(objs) == 1
        protein = objs[0]
    elif pm.get_object_list("protein"):
        protein = "protein"
    pm.set_name(protein, f"{group}.protein")
    pm.group(group, f"{group}.protein")

    clusters, eclusters = get_clusters()
    k15_list = get_kozakov2015(group, clusters, k15_max_length)
    if run_fpocket:
        fpo_list = get_fpocket(group, f"{group}.protein")
    process_clusters(group, clusters)
    process_eclusters(group, eclusters)
    if run_fpocket:
        e19_list = get_egbert2019(group, fpo_list, clusters)
        k15_dl_list = get_kozakov2015_large(group, fpo_list, clusters)

    pm.hide("everything", f"{group}.*")

    pm.show("cartoon", f"{group}.protein")
    pm.show("mesh", f"{group}.K15_D* or {group}.K15_B*")

    pm.show("spheres", f"{group}.ACS_*")
    pm.set("sphere_scale", 0.25, f"{group}.ACS_*")

    pm.color("red", f"{group}.K15_D*")
    pm.color("salmon", f"{group}.K15_B*")
    pm.color("red", f"{group}.ACS_acceptor_*")
    pm.color("blue", f"{group}.ACS_donor_*")
    pm.color("green", f"{group}.ACS_halogen_*")
    pm.color("orange", f"{group}.ACS_aromatic_*")
    pm.color("yellow", f"{group}.ACS_apolar_*")

    pm.show("line", f"{group}.CS_*")
    pm.show("spheres", f"{group}.fpocket_*")
    pm.set("sphere_scale", 0.25, f"{group}.fpocket_*")
    pm.color("white", f"{group}.fpocket_*")

    pm.disable(f"{group}.CS_*")
    pm.disable(f"{group}.fpocket_*")
    
    pm.set("mesh_mode", 1)
    pm.orient("all")

    pm.order(f"{group}.fpocket_*", location="top")
    pm.order(f"{group}.K15_*", location="top")
    pm.order(f"{group}.protein", location="top")

    ret = SimpleNamespace(
        clusters=clusters,
        eclusters=eclusters,
        kozakov2015=k15_list,
    )
    if run_fpocket:
        ret.kozakov2015 = [*ret.kozakov2015, *k15_dl_list]
        ret.egbert2019 = e19_list
        ret.fpocket = fpo_list
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


@declare_command
def fo(
    sel1: Selection,
    sel2: Selection,
    state1: int = 1,
    state2: int = 1,
    radius: float = 2,
    verbose: bool = True,
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

    EXAMPLE
        fo REF_LIG, ftmap1234.D_003_*_*
    """
    atoms1 = pm.get_coords(f"({sel1}) and not elem H", state=state1)
    atoms2 = pm.get_coords(f"({sel2}) and not elem H", state=state2)
    if atoms1 is None or atoms2 is None:
        fo_ = 0
    else:
        dist = distance_matrix(atoms1, atoms2) - radius <= 0
        num_contacts = np.sum(np.any(dist, axis=1))
        total_atoms = len(atoms1)
        fo_ = num_contacts / total_atoms
    if verbose:
        print(f"FO: {fo_:.2f}")
    return fo_


@declare_command
def dc(
    sel1: Selection,
    sel2: Selection,
    state1: int = 1,
    state2: int = 1,
    radius: float = 1.25,
    verbose: bool = True,
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
        verbose define verbosity

    EXAMPLES
        dc REF_LIG, ftmap1234.D_003_*_*
        dc ftmap1234.D.003, REF_LIG, radius=1.5

    """
    xyz1 = pm.get_coords(f"({sel1}) and not elem H", state=state1)
    xyz2 = pm.get_coords(f"({sel2}) and not elem H", state=state2)

    dc_ = (distance_matrix(xyz1, xyz2) < radius).sum()
    if verbose:
        print(f"DC: {dc_:.2f}")
    return dc_


@declare_command
def dce(
    sel1: Selection,
    sel2: Selection,
    state1: int = 1,
    state2: int = 1,
    radius: float = 1.25,
    verbose: bool = True,
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
        verbose define verbosity (default: true)

    EXAMPLE
        dce REF_LIG, ftmap1234.D_003_*_*
    """
    dc_ = dc(sel1, sel2, radius=radius, state1=state1, state2=state2, verbose=False)
    dce_ = dc_ / pm.count_atoms(f"({sel1}) and not elem H")
    if verbose:
        print(f"DCE: {dce_:.2f}")
    return dce_


class LinkageMethod(StrEnum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"


@declare_command
def fp_sim(
    exprs: Selection,
    site: str = "*",
    radius: int = 4,
    nbins: int = 5,
    axis_fingerprint: str = '',
    axis_dendrogram: str = '',
    linkage_method: LinkageMethod = LinkageMethod.WARD,
    quiet: bool = True,
):
    """
    Compute the similarity between the residue contact fingerprint of two
    hotspots.

    OPTIONS:
        hotspots          hotspot expressions
        site              selection to focus based on first protein
        radius            radius to compute the contacts (default: 4)
        plot_fingerprints plot the fingerprints (default: True)
        nbins             number of residue labels (default: 5)
        plot_dendrogram   plot the dendrogram (default: False)
        linkage_method    linkage method (default: single)
        verbose           define verbosity

    EXAMPLES
        fs_sim 8DSU.K15_D_01* 6XHM.K15_D_01*
        fs_sim 8DSU.CS_* 6XHM.CS_*, site=resi 8-101, nbins=10
    """

    # Identify objects/hotspots to probe count    
    hotspots = []
    groups = set()
    for object_set in multiple_expression_selector(exprs):
        for object in object_set:
            if group := pm.get_property('Group', object):
                hotspots.append(object)
                groups.add(group)

    proteins = [f"{g}.protein" for g in groups]
    mapping = get_mapping(
        proteins[0],
        '*.protein',
        transform=False,
        site=site,
        radius=radius
    )
    fpts = []
    for hs in hotspots:
        fpt = {}
        @mapping.groupby(['chain', 'resi'], as_index=False).apply
        def apply(group):
            for idx, row in group.iterrows():
                resn = ONE_LETTER.get(row.resn, "X")
                lbl = (resn, row.resi, row.chain)
                cnt = pm.count_atoms(
                    f"(%{hs}) within {radius} from (byres %{row.model} & index {row['index']})"
                )
                fpt[lbl] = fpt.get(lbl, 0) + cnt
        fpts.append(fpt)
        
        fpt0 = fpts[0]
        if not all([len(fpt0) == len(fpt) for fpt in fpts]):
            raise ValueError(
                "All fingerprints must have the same length. "
                "Do you have incomplete structures?"
            )
        
        with mpl_axis(axis_fingerprint, nrows=len(hotspots), sharey=True) as axs:
            for ax, fpt, hs in zip(axs, fpts, hotspots):
                labels = ['%s %s:%s' % k for k in fpt]
                arange = np.arange(len(fpt))
                ax.bar(arange, fpt.values())
                ax.set_title(hs)
                ax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
                ax.set_xticks(arange, labels=labels, rotation=90)
                ax.locator_params(axis="x", tight=True, nbins=nbins)
                for label in ax.xaxis.get_majorticklabels():
                    label.set_horizontalalignment("right")

    with mpl_axis(axis_dendrogram) as ax:  
        corrs = []
        labels = []
        for i1, (fp1, hs1) in enumerate(zip(fpts, hotspots)):
            labels.append(hs1)
            for i2, (fp2, hs2) in enumerate(zip(fpts, hotspots)):
                if i1 >= i2:
                    continue
                corr = pearsonr(list(fp1.values()), list(fp2.values())).statistic
                if np.isnan(corr):
                    corr = 0
                corrs.append(corr)
                if not quiet:
                    print(f"Pearson correlation: {hs1} / {hs2}: {corr:.2f}")

        dendrogram(
            [1 - c for c in corrs],
            method=linkage_method,
            labels=labels,
            ax=ax,
            leaf_rotation=90,
            color_threshold=0,
        )   
        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment("right")
    return fpts, corrs, labels



@declare_command
def ho(
    hs1: Selection,
    hs2: Selection,
    radius: float = 2.5,
    verbose: bool = True,
):
    """
    Compute the Hotspot Overlap (HO) metric. HO is defined as the number of
    atoms in hs1 in contact with hs2 plus the number of atoms in hs2 in
    contact with hs1 divided by the total number of atoms in both hotspots.

    OPTIONS
        hs1     an hotspot object
        hs2     another hotspot object
        radius  the distance to consider two atoms in contact (default: 2.5)
        verbose define verbosity
    """
    atoms1 = pm.get_coords(f"({hs1}) and not elem H")
    atoms2 = pm.get_coords(f"({hs2}) and not elem H")
    dist = distance_matrix(atoms1, atoms2) - radius <= 0
    num_contacts1 = np.sum(np.any(dist, axis=1))
    num_contacts2 = np.sum(np.any(dist, axis=0))
    ho = (num_contacts1 + num_contacts2) / (len(atoms1) + len(atoms2))
    if verbose:
        print(f"HO: {ho:.2f}")
    return ho


class ResidueSimilarityMethod(StrEnum):
    JACCARD = "jaccard"
    OVERLAP = "overlap"


@declare_command
def res_sim(
    hs1: Selection,
    hs2: Selection,
    radius: float = 2,
    align: bool = False,
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
        verbose define verbosity

    EXAMPLES
        res_sim 8DSU.D_001*, 6XHM.D_001*
        res_sim 8DSU.CS_*, 6XHM.CS_*
    """
    try:
        group1 = pm.get_property('Group', hs1)
    except:
        group1 = hs1
    try:
        group2 = pm.get_property('Group', hs2)
    except:
        group2 = hs2

    sel1 = f"{group1}.protein within {radius} from ({hs1})"
    sel2 = f"{group2}.protein within {radius} from ({hs2})"

    resis1 = set()
    pm.iterate(sel1, "resis1.add((chain, resi))", space={"resis1": resis1})

    if group1 == group2 or not align:
        resis2 = set()
        pm.iterate(sel2, "resis2.add((chain, resi))", space={"resis2": resis2})
    else:
        try:
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
        print("Your selection yields zero atoms.")
        return 0.0

    if not quiet:
        print(f"{method} similarity: {ret:.2}")
    return ret


class EFtmapOverlapType(StrEnum):
    donor = "donor"
    acceptor = "acceptor"
    apolar = "apolar"
    aromatic = "aromatic"


def _eftmap_overlap_get_aromatic(lig):
    lig_model = pm.get_model(lig)
    aromatic = set()
    for bond in lig_model.bond:
        if bond.order == 4:   # This is how ChemPy detect aromatic bonds
            aromatic.update(bond.index)
    xyz = []
    for idx in aromatic:
        atom = lig_model.atom[idx]
        xyz.append(atom.coord)
    return np.array(xyz)


def _eftmap_overlap_get_donor(lig):
    return pm.get_coords(f"({lig}) and donor")


def _eftmap_overlap_get_acceptor(lig):
    return pm.get_coords(f"({lig}) and acceptor")


def _eftmap_overlap_get_apolar(lig):
    return pm.get_coords(f"({lig}) and elem C")


def _eftmap_overlap_get_halogen(lig):
    return pm.get_coords(f"({lig}) and elem F+Br+Cl+I")


def eftmap_overlap(lig, hs, radius=2):
    if '.ACS_donor_' in hs:
        lig_xyz = _eftmap_overlap_get_donor(lig)
    elif '.ACS_acceptor_' in hs:
        lig_xyz = _eftmap_overlap_get_acceptor(lig)
    elif '.ACS_apolar_' in hs:
        lig_xyz = _eftmap_overlap_get_apolar(lig)
    elif '.ACS_halogen' in hs:
        lig_xyz = _eftmap_overlap_get_halogen(lig)
    elif '.ACS_aromatic' in hs:
        lig_xyz = _eftmap_overlap_get_aromatic(lig)
    else:
        raise RuntimeError(f"Unknown hotspot type: {hs}")
    hs_xyz = pm.get_coords(hs)
    dist = distance_matrix(lig_xyz, hs_xyz)
    contacts = np.any(dist - radius <= 0, axis=1)
    return np.sum(contacts)


class HeatmapFunction(StrEnum):
    HO = "ho"
    RESIDUE_JACCARD = "residue_jaccard"
    RESIDUE_OVERLAP = "residue_overlap"


@declare_command
def plot_heatmap(
    exprs: Selection,
    method: HeatmapFunction = HeatmapFunction.HO,
    radius: float = 2.0,
    align: bool = False,
    axis: str = ''
):
    """
    Compute the similarity between matching objects using a similarity function.

    OPTIONS
        objs        space separated list of object expressions
        method      ho, residue_jaccard, or residue_overlap (default: ho)
        radius      the radius to consider atoms in contact (default: 2.0)
        annotate    fill the cells with values

    EXAMPLES
        cross_measure *.D_000_*_*, function=residue_jaccard
        cross_measure *.D_*. align=True
        cross_measure *.D_000_*_* *.DS_*
    """
    def sort(obj):
        klass = pm.get_property("Class", obj)
        return str(klass), obj

    objects = expression_selector(exprs)
    objects = list(sorted(objects, key=sort))
    
    mat = []
    for idx1, obj1 in enumerate(objects):
        mat.append([])
        for idx2, obj2 in enumerate(objects):
            if idx1 == idx2:
                ret = 1
            elif idx1 < idx2:
                ret = np.nan
            else:
                match method:
                    case HeatmapFunction.HO:
                        ret = ho(obj1, obj2, radius=radius, verbose=False)
                    case HeatmapFunction.RESIDUE_JACCARD:
                        ret = res_sim(
                            obj1,
                            obj2,
                            radius=radius,
                            method="jaccard",
                            align=align,
                        )
                    case HeatmapFunction.RESIDUE_OVERLAP:
                        ret = res_sim(
                            obj1,
                            obj2,
                            radius=radius,
                            method="overlap",
                            align=align,
                        )
            mat[-1].append(round(ret, 2))
    
    
    with mpl_axis(axis) as ax:
        mat_masked = np.ma.masked_invalid(mat)
        ax.imshow(mat_masked, cmap='viridis')
        plt.xticks(np.arange(len(mat)), objects, rotation=90)
        plt.yticks(np.arange(len(mat)), objects)    
    return mat, objects


class PrioritizationType(StrEnum):
    RESIDUE = "residue"
    ATOM = "atom"


@declare_command
def hs_proj(
    sel: Selection,
    protein: Selection = "",
    radius: int = 4,
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


@declare_command
def plot_hca(
    exprs: Selection,
    residue_radius: float = 4.0,
    residue_align: bool = False,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: bool = -1,
    axis: str = None
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

    def _get_property_vector(hs_type, obj):
        x, y, z = np.mean(pm.get_coords(obj), axis=0)

        if hs_type == "K15":
            S = pm.get_property("S", obj)
            S0 = pm.get_property("S0", obj)
            CD = pm.get_property("CD", obj)
            MD = pm.get_property("MD", obj)
            return np.array([S, S0, CD, MD, x, y, z])
        elif hs_type == "CS":
            S = pm.get_property("S", obj)
            return np.array([S, x, y, z])
        elif hs_type == "ACS":
            S = pm.get_property("S", obj)
            MD = pm.get_property("MD", obj)
            return np.array([S, MD, x, y, z])

    def _euclidean_like(hs_type, p1, p2, j):
        if hs_type == "K15":
            return np.sqrt(
                (p1[0] - p2[0]) ** 2
                + (p1[1] - p2[1]) ** 2
                + (p1[2] - p2[2]) ** 2
                + (p1[3] - p2[3]) ** 2
                + (p1[4] - p2[4]) ** 2
                + (p1[5] - p2[5]) ** 2
                + (p1[6] - p2[6]) ** 2
                + (1 - j) ** 2
            )
        elif "CS":
            return np.sqrt(
                (p1[0] - p2[0]) ** 2
                + (p1[1] - p2[1]) ** 2
                + (p1[2] - p2[2]) ** 2
                + (p1[3] - p2[3]) ** 2
                + (1 - j) ** 2
            )
        elif hs_type == "ACS":
            return np.sqrt(
                (p1[0] - p2[0]) ** 2
                + (p1[1] - p2[1]) ** 2
                + (p1[2] - p2[2]) ** 2
                + (p1[3] - p2[3]) ** 2
                + (p1[4] - p2[4]) ** 2
                + (1 - j) ** 2
            )

    object_list = list(expression_selector(exprs))
    assert len(set(pm.get_property("Type", o) for o in object_list)) == 1, object_list
    
    hs_type = pm.get_property("Type", object_list[0])
    if hs_type == "K15":
        n_props = 4
    elif hs_type == "CS":
        n_props = 1
    elif hs_type == "ACS":
        n_props = 2
    labels = []
    p = np.zeros((len(object_list), n_props + 3))

    for idx, obj in enumerate(object_list):
        p[idx, :] = _get_property_vector(hs_type, obj)
        labels.append(obj)

    for col in range(n_props + 3):
        if max(p[:, col]) - min(p[:, col]) == 0:
            p[:, col] = 0
        else:
            p[:, col] = (p[:, col] - min(p[:, col])) / (max(p[:, col]) - min(p[:, col]))

    X = []
    for idx1, obj1 in enumerate(object_list):
        for idx2, obj2 in enumerate(object_list):
            if idx1 >= idx2:
                continue
            p1 = p[idx1, :]
            p2 = p[idx2, :]
            if residue_align:
                j = res_sim(
                    obj1,
                    obj2,
                    radius=residue_radius,
                    align=residue_align,
                )
            else:
                j = 0
            d = _euclidean_like(hs_type, p1, p2, j)
            X.append(d)
    return plot_hca_base(X, labels, linkage_method, color_threshold, axis)



#
# GRAPHICAL USER INTERFACE
#

from pymol import Qt

QWidget = Qt.QtWidgets.QWidget
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

QtCore = Qt.QtCore
QIcon = Qt.QtGui.QIcon


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

        self.maxLengthSpin = QSpinBox()
        self.maxLengthSpin.setValue(5)
        self.maxLengthSpin.setMinimum(3)
        self.maxLengthSpin.setMaximum(8)
        boxLayout.addRow("Max length:", self.maxLengthSpin)

        self.runFpocketCheck = QCheckBox()
        self.runFpocketCheck.setChecked(False)
        boxLayout.addRow("Run Fpocket:", self.runFpocketCheck)

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

    def clearRows(self):
        self.table.setRowCount(0)

    def load(self):
        max_length = self.maxLengthSpin.value()
        try:
            for row in range(self.table.rowCount()):
                group = self.table.item(row, 0).text()
                filename = self.table.item(row, 1).text()
                run_fpocket = self.runFpocketCheck.isChecked()
                try:
                    load_ftmap(
                        filename,
                        group=group,
                        k15_max_length=max_length,
                        run_fpocket=run_fpocket,
                    )
                except Exception:
                    try:
                        load_ftmap(
                            filename,
                            group=group,
                            k15_max_length=max_length,
                            run_fpocket=run_fpocket,

                        )
                    except Exception:
                        if not os.path.exists(filename):
                            raise ValueError(f"File does not exist: '{filename}'")
                        else:
                            raise Exception(f"Failed to load file: '{filename}'")
        finally:
            self.clearRows()


class SortableItem(QTableWidgetItem):
    def __init__(self, obj):
        super().__init__(str(obj))
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEditable)

    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()


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

            @self.itemClicked.connect
            def itemClicked(item):
                obj = self.item(item.row(), 0).text()
                pm.select(obj)
                pm.enable("sele")

        def hideEvent(self, evt):
            self.clearSelection()

    def __init__(self):
        super().__init__()
        self.selected_objs = set()
        self.current_tab = 'K15'
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        filter_line = QLineEdit()
        layout.addWidget(filter_line)

        @filter_line.textEdited.connect
        def textEdited(expr):
            if not expr.strip():
                return
            self.selected_objs = expression_selector(expr, self.current_tab)
            self.refresh()
        tab = QTabWidget()
        layout.addWidget(tab)
        
        self.hotspotsMap = {
            ("Kozakov2015", "K15"): ["Class", "S", "S0", "CD", "MD", "Length", "Fpocket"],
            ("CS", "CS"): ["S"],
            ("ACS", "ACS"): ["Class", "S", "MD"],
            ("Egbert2019", "E19"): ["Fpocket", "S","S0", "S1", "Length"],
            ("Fpocket", "Fpocket"): ["Pocket Score", "Drug Score"],
        }

        @tab.currentChanged.connect
        def currentChanged(tab_index):
            self.current_tab = [k[1] for k in self.hotspotsMap.keys()][tab_index]
            expr = filter_line.text()
            self.selected_objs = expression_selector(expr, self.current_tab)
            self.refresh()
            
        self.tables = {}
        for (title, key), props in self.hotspotsMap.items():
            table = self.TableWidgetImpl(props)
            self.tables[title] = table
            tab.addTab(table, title)

        exportButton = QPushButton(QIcon("save"), "Export Tables")
        exportButton.clicked.connect(self.export)
        layout.addWidget(exportButton)

    def showEvent(self, event):
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
                if not pm.get_property_list(obj):
                    continue
                obj_type = pm.get_property('Type', obj)
                if obj_type == key:
                    if len(self.selected_objs) == 0:
                        self.appendRow(title, key, obj)
                    elif obj in self.selected_objs:
                        self.appendRow(title, key, obj)

            self.tables[title].setSortingEnabled(True)

    def appendRow(self, title, key, obj):
        self.tables[title].insertRow(self.tables[title].rowCount())
        line = self.tables[title].rowCount() - 1

        self.tables[title].setItem(line, 0, SortableItem(obj))

        for idx, prop in enumerate(self.hotspotsMap[(title, key)]):
            prop_value = pm.get_property(prop, obj)
            self.tables[title].setItem(line, idx + 1, SortableItem(prop_value))

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
                for (title, key), props in self.hotspotsMap.items():
                    data = {"Object": [], **{p: [] for p in props}}
                    for header in data:
                        column = list(data.keys()).index(header)
                        for line in range(self.tables[title].rowCount()):
                            item = self.tables[title].item(line, column)
                            data[header].append(self.parse_item(item))
                    df = pd.DataFrame(data)
                    df.to_excel(xlsx_writer, sheet_name=title, index=False)

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


class SimilarityWidget(QWidget):

    def __init__(self):
        super().__init__()

        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        groupBox = QGroupBox("General")
        mainLayout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.hotspotExpressionLine = QLineEdit()
        boxLayout.addRow("Hotspots:", self.hotspotExpressionLine)

        layout = QHBoxLayout()
        mainLayout.addLayout(layout)

        groupBox = QGroupBox("Heatmap")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.functionCombo = QComboBox()
        self.functionCombo.addItems([e.value for e in HeatmapFunction])
        boxLayout.addRow("Function:", self.functionCombo)

        self.radiusSpin = QSpinBox()
        self.radiusSpin.setValue(2)
        self.radiusSpin.setMinimum(1)
        self.radiusSpin.setMaximum(5)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.heatmapAlignCheck = QCheckBox()
        self.heatmapAlignCheck.setChecked(False)
        boxLayout.addRow("Align:", self.heatmapAlignCheck)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_heatmap)
        boxLayout.addWidget(plotButton)

        groupBox = QGroupBox("HCA")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.residueRadiusSpin = QDoubleSpinBox()
        self.residueRadiusSpin.setValue(4.0)
        self.residueRadiusSpin.setMinimum(0)
        self.residueRadiusSpin.setMaximum(10)
        boxLayout.addRow("Residue radius:", self.residueRadiusSpin)

        self.resiudeAlignCheck = QCheckBox()
        self.resiudeAlignCheck.setChecked(False)
        boxLayout.addRow("Residue Align:", self.resiudeAlignCheck)

        self.linkageMethodCombo = QComboBox()
        self.linkageMethodCombo.addItems([e.value for e in LinkageMethod])
        boxLayout.addRow("Linkage:", self.linkageMethodCombo)

        self.colorThresholdSpin = QDoubleSpinBox()
        self.colorThresholdSpin.setMinimum(-0.001)
        self.colorThresholdSpin.setValue(-0.001)
        self.colorThresholdSpin.setSingleStep(0.001)
        self.colorThresholdSpin.setDecimals(2)
        boxLayout.addRow("Color threshold:", self.colorThresholdSpin)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_dendrogram)
        boxLayout.addWidget(plotButton)

    def plot_heatmap(self):
        expression = self.hotspotExpressionLine.text()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        align = self.heatmapAlignCheck.isChecked()
        plot_heatmap(expression, function, radius, align)

    def plot_dendrogram(self):
        expression = self.hotspotExpressionLine.text()
        residue_radius = self.residueRadiusSpin.value()
        residue_align = self.resiudeAlignCheck.isChecked()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()

        return plot_hca(
            expression,
            residue_radius,
            residue_align,
            linkage_method,
            color_threshold,
        )


class CountWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        self.setLayout(layout)

        groupBox = QGroupBox("Residue projection")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.hotspotsExpressionLine1 = QLineEdit()
        boxLayout.addRow("Hotspots:", self.hotspotsExpressionLine1)

        self.proteinExpressionLine = QLineEdit()
        boxLayout.addRow("Protein:", self.proteinExpressionLine)

        self.radiusSpin = QSpinBox()
        self.radiusSpin.setValue(3)
        self.radiusSpin.setMinimum(2)
        self.radiusSpin.setMaximum(5)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.typeCombo = QComboBox()
        self.typeCombo.addItems([e.value for e in PrioritizationType])
        boxLayout.addRow("Type:", self.typeCombo)

        self.paletteLine = QLineEdit("rainbow")
        boxLayout.addRow("Palette:", self.paletteLine)

        drawButton = QPushButton("Draw")
        drawButton.clicked.connect(self.draw_projection)
        boxLayout.addWidget(drawButton)

        groupBox = QGroupBox("Fingerprint vector")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.hotspotsExpressionLine2 = QLineEdit("")
        boxLayout.addRow("Hotspots:", self.hotspotsExpressionLine2)

        self.siteExpressionLine = QLineEdit("*")
        boxLayout.addRow("Site:", self.siteExpressionLine)

        self.radiusSpin = QSpinBox()
        self.radiusSpin.setValue(4)
        self.radiusSpin.setMinimum(2)
        self.radiusSpin.setMaximum(5)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.nBinsSpin = QSpinBox()
        self.nBinsSpin.setValue(5)
        self.nBinsSpin.setMinimum(0)
        self.nBinsSpin.setMaximum(100)
        boxLayout.addRow("Fingerprint bins:", self.nBinsSpin)

        self.fingerprintsCheck = QCheckBox()
        self.fingerprintsCheck.setChecked(True)
        boxLayout.addRow("Fingerprints:", self.fingerprintsCheck)

        self.dendrogramCheck = QCheckBox()
        self.dendrogramCheck.setChecked(False)
        boxLayout.addRow("Dendrogram:", self.dendrogramCheck)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_fingerprint)
        boxLayout.addWidget(plotButton)

    def draw_projection(self):
        hotspots = self.hotspotsExpressionLine1.text()
        protein = self.proteinExpressionLine.text()
        radius = self.radiusSpin.value()
        type = self.typeCombo.currentText()
        palette = self.paletteLine.text()

        hs_proj(hotspots, protein, radius, type, palette)

    def plot_fingerprint(self):
        hotspots = self.hotspotsExpressionLine2.text()
        site = self.siteExpressionLine.text()
        radius = self.radiusSpin.value()
        fingerprints = self.fingerprintsCheck.isChecked()
        dendrogram = self.dendrogramCheck.isChecked()
        nbins = self.nBinsSpin.value()

        fp_sim(
            hotspots,
            site,
            radius,
            verbose=True,
            plot_fingerprints=fingerprints,
            plot_dendrogram=dendrogram,
            nbins=nbins,
        )


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(600, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("XDrugPy")

        tab = QTabWidget()
        tab.addTab(LoadWidget(), "Load")
        tab.addTab(TableWidget(), "Properties")
        tab.addTab(SimilarityWidget(), "Hotspot Similarity")
        tab.addTab(CountWidget(), "Probe Count")

        layout.addWidget(tab)


dialog = None


def run_plugin_gui():
    global dialog
    if dialog is None:
        dialog = MainDialog()
    dialog.show()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("XDrugPy::Hotspots", run_plugin_gui)
