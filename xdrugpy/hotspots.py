import os.path
import shutil
import subprocess
import re
import tempfile
from glob import glob
from itertools import combinations
from types import SimpleNamespace
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
from scipy.spatial import distance_matrix, distance
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from strenum import StrEnum

from .utils import (
    declare_command,
    Selection,
    plot_hca_base,
    clustal_omega
)

from pymol import cmd as pm

matplotlib.use("Qt5Agg")


def get_clusters():
    clusters = []
    eclusters = []

    for obj in pm.get_object_list():
        if obj.startswith(f"crosscluster."):
            _, _, s, _ = obj.split(".", maxsplit=4)
            coords = pm.get_coords(f"%{obj} & !elem H")
            clusters.append(
                SimpleNamespace(
                    selection=obj,
                    strength=int(s),
                    coords=coords,
                )
            )
        elif obj.startswith("consensus."):
            _, _, s = obj.split(".", maxsplit=3)
            coords = pm.get_coords(f"%{obj} & !elem H")
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
            averages = [np.average(c.coords, axis=0) for c in combination]
            cd = [distance.euclidean(averages[0], avg) for avg in averages]

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

    unwanted = set()
    for ix1, hs1 in enumerate(k15):
        for ix2, hs2 in enumerate(k15):
            if hs1.kozakov_class != hs2.kozakov_class:
                continue
            if ix1 >= ix2:
                continue
            if get_fo(hs1.selection, hs2.selection) == 1:
                unwanted.add(hs1.selection)
    for hs_sele in unwanted:
        k15 = list(filter(
            lambda hs: hs.selection != hs_sele,
            k15
        ))
        pm.delete(hs.selection)

    idx = 0
    cur_class = None
    for hs in k15:
        if hs.kozakov_class != cur_class:
            cur_class = hs.kozakov_class
            idx = -1
        idx += 1
        new_name = f"{group}.K15_{cur_class}_{idx:02}"
        pm.create(new_name, hs.selection)
        pm.group(group, new_name)
        hs.selection = new_name
        set_properties(
            hs,
            new_name,
            {
                "Type": "K15",
                "Group": group,
                "Selection": new_name,
                "Class": hs.kozakov_class,
                "ST": hs.strength,
                "S0": hs.strength0,
                "CD": round(hs.center_center, 2),
                "MD": round(hs.max_dist, 2),
                "Length": hs.length,
            },
        )
    return k15


def get_fpocket(group, protein):
    pockets = []
    # FIXME TODO
    with tempfile.TemporaryDirectory() as tempdir:
        protein_pdb = f"{tempdir}/{group}.pdb"
        pm.save(protein_pdb, selection=protein)
        subprocess.check_output(
            [shutil.which("fpocket"), "-f", protein_pdb],
        )
        header_re = re.compile(r"^HEADER\s+\d+\s+-(.*):(.*)$")
        for pocket_pdb in glob(f"{tempdir}/{group}_out/pockets/pocket*_atm.pdb"):
            idx = (
                os.path.basename(pocket_pdb)
                .replace("pocket", "")
                .replace("_atm.pdb", "")
            )
            idx = int(idx)
            pocket = SimpleNamespace(selection=f"{group}.fpocket_{idx:02}")
            pm.delete(pocket.selection)
            pm.load(pocket_pdb, pocket.selection)
            set_properties(
                pocket, pocket.selection, {
                    "Type": "Fpocket",
                    "Group": group,
                    "Selection": pocket.selection,
                }
            )
            for line in pm.get_property("pdb_header", pocket.selection).split("\n"):
                if match := header_re.match(line):
                    prop = match.group(1).strip()
                    value = float(match.group(2))
                    set_properties(pocket, pocket.selection, {
                        prop: value
                    })
            pockets.append(pocket)
    return pockets


def process_clusters(group, clusters):
    for idx, cs in enumerate(clusters):
        new_name = f"{group}.CS_{idx:02}"
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
                "ST": cs.strength,
            },
        )
    pm.delete("consensus.*")
    pm.delete("crosscluster.*")


def process_eclusters(group, eclusters):
    for acs in eclusters:
        new_name = f"{group}.ACS_{acs.probe_type}_{acs.idx:02}"
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
                "ST": acs.strength,
                "MD": round(md, 2),
            },
        )
    pm.delete("clust.*")


def get_egbert2019(group, fpo_list):
    e19 = []
    ix_e19 = 0
    for pocket in fpo_list:
        sel = f"{group}.CS_* near_to 3 of {pocket.selection}"
        objs = pm.get_object_list(sel)
        s_list = [pm.get_property("ST", o) for o in objs]
        if len(objs) >= 4 and sum([s >= 16 for s in s_list]) >= 2:
            new_name = f"{group}.E19_{ix_e19:02}"
            pm.create(new_name, sel)
            pm.group(group, new_name)
            hs = SimpleNamespace()
            set_properties(hs, new_name, {
                "Type": "E19",
                "Group": group,
                "Selection": new_name,
                "Fpocket": pocket.selection,
                "ST": sum(s_list),
                "S0": s_list[0],
                "S1": s_list[1],
                "S2": s_list[2],
                "S3": s_list[3],
                "Length": len(objs),
            })
            e19.append(hs)
            ix_e19 += 1
    return e19


def get_kozakov2015_large(group, fpo_list, clusters):
    k15l = []
    idx = 0
    for pocket in fpo_list:
        sel = f"byobject ({group}.CS_* within 3 of {pocket.selection})"
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

        if 13 <= s0 < 16 and cd >= 8 and md >= 10 and sz >= 5:
            new_name = f"{group}.K15_BL_{idx:02}"
            klass = "BL"
        elif s0 >= 16 and cd >= 8 and md >= 10 and sz >= 5:
            new_name = f"{group}.K15_DL_{idx:02}"
            klass = "DL"
        else:
            continue
        idx += 1

        hs = SimpleNamespace()
        hs.kozakov_class = klass
        pm.create(new_name, sel)
        pm.group(group, new_name)
        set_properties(
            hs,
            new_name,
            {
                "Type": "K15",
                "Group": group,
                "Selection": new_name,
                "Class": klass,
                "ST": strength,
                "S0": s0,
                "CD": round(cd, 2),
                "MD": round(md, 2),
                "Length": len(pocket_clusters),
                "Fpocket": pocket.selection,
            },
        )
        hs.selection = hs.Selection
        if hs.kozakov_class:
            k15l.append(hs)

    return k15l

class FtmapResults:
    pass

@declare_command
def load_ftmap(
    filenames: List[Path] | Path,
    groups: List[str] | str = "",
    k15_max_length: int = 3,
    run_fpocket: bool = False,
    bekar_label: bool = '',
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
                rets.append(_load_ftmap(fnames, groups, k15_max_length, run_fpocket))
            except:
                rets.append(_load_ftmap(fnames, groups, k15_max_length, run_fpocket))
        if single:
            return rets[0]
        else:
            ftmap = FtmapResults()
            ftmap.results = rets
            if bekar_label:
                bekar, cs16_count, k15d_count = eval_bekar25_limits(bekar_label, rets)
                ftmap.bekar25 = bekar
                ftmap.cs16_count = cs16_count
                ftmap.k15d_count = k15d_count
            else:
                ftmap.bekar25 = None
                ftmap.cs16_count = None
                ftmap.k15d_count = None
            return ftmap
    finally:
        pm.set('defer_updates', 0)


def eval_bekar25_limits(label, ftmap_results):   
    cs16_count = 0
    k15d_count = 0
    for res in ftmap_results:
        for cs in res.clusters:
            if cs.strength >= 16:
                cs16_count += 1
                break
            elif cs.strength <= 15:
                continue
        for k15 in res.kozakov2015:
            if k15.kozakov_class in ["D", "DS", "DL"]:
                k15d_count += 1
                break
            elif k15.kozakov_class in ["B", "BS", "BL"]:
                continue
    bekar = cs16_count / len(ftmap_results) > 0.7 and k15d_count / len(ftmap_results) > 0.5
    obj = SimpleNamespace()
    obj_name = f"_BC25_{label}"
    pm.pseudoatom(obj_name)
    set_properties(
        obj,
        obj_name,
        {
            'Type': 'BC25',
            'Total': len(ftmap_results),
            'CS16': cs16_count,
            'K15D': k15d_count,
            'IsBekar': bekar,
        }
    )
    return bekar, cs16_count, k15d_count


def _load_ftmap(
    filename: Path, group: str = "", k15_max_length: int = 3, run_fpocket: bool = False
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
    group = pm.get_legal_name(group)
    
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
        e19_list = get_egbert2019(group, fpo_list)
        k15_dl_list = get_kozakov2015_large(group, fpo_list, clusters)

    pm.hide("everything", f"{group}.*")

    pm.show("cartoon", f"{group}.protein")
    pm.show("mesh", f"{group}.K15_D* or {group}.K15_B* or {group}.E19_*")

    pm.show("spheres", f"{group}.ACS_*")
    pm.set("sphere_scale", 0.25, f"{group}.ACS_*")

    pm.color("red", f"{group}.K15_D*")
    pm.color("salmon", f"{group}.K15_B*")
    pm.color("raspberry", f"{group}.E19_*")

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
def get_fo(
    sel1: Selection,
    sel2: Selection,
    radius: float = 2,
    quiet: int = True,
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

    dist = distance_matrix(xyz1, xyz2) <= radius
    nc = np.sum(np.any(dist, axis=1))
    nt = len(xyz1)
    fo = nc / nt
    if not quiet:
        print(f"FO: {fo:.2f}")
    return fo


@declare_command
def get_dc(
    sel1: Selection,
    sel2: Selection,
    radius: float = 1.25,
    quiet: int = True,
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
    
    dc = (distance_matrix(xyz1, xyz2) < radius).sum()
    if not quiet:
        print(f"DC: {dc:.2f}")
    return dc


@declare_command
def get_dce(
    sel1: Selection,
    sel2: Selection,
    radius: float = 1.25,
    quiet: int = True,
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


@declare_command
def fpt_sim(
    multi_seles: Selection,
    site: Selection = "*",
    site_radius: float = 4.0,
    conservation: str = "*:.",
    contact_radius: float = 3.0,
    nbins: int = 5,
    sharex: bool = True,
    linkage_method: LinkageMethod = LinkageMethod.WARD,
    color_threshold: float = 0.0,
    hide_threshold: bool = False,
    annotate: bool = True,
    plot_fingerprints: str = "",
    plot_hca: str = "",
    quiet: int = True,
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
        if group := pm.get_property("Group", obj):
            groups.append(group)
    polymers = [f"{g}.protein" for g in groups]
    assert len(polymers) > 0, "Please review your selections"

    ref_polymer = polymers[0]
    ref_sele = f"{ref_polymer} and ({ref_polymer} within {site_radius} of ({site}))"
    site_resis = []
    for at in pm.get_model(f"({ref_sele}) & guide & polymer").atom:
        site_resis.append((at.model, at.index))
    
    mapping = clustal_omega(list(polymers), conservation)
    ref_map = mapping[ref_polymer]
    fpts = []
    for hs, (poly, map) in zip(seles, mapping.items()):
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
        if not isinstance(axs, (np.ndarray, list)):
            axs = [axs]
        if not all([len(fpts[0]) == len(fpt) for fpt in fpts]):
            raise ValueError(
                "All fingerprints must have the same length. "
                "Do you have incomplete structures?"
            )
        
        for ix, (ax, fpt, sele) in enumerate(zip(axs, fpts, seles)):
            labels = ["%s%s %s_%s" % k for k in fpt]
            if sharex and ix == 0:
                shared_labels = labels
            elif sharex and ix + 1 == len(seles):
                labels = shared_labels
            arange = np.arange(len(fpt))
            ax.bar(arange, fpt.values(), color="C0")
            ax.set_title(sele)
            ax.yaxis.set_major_formatter(lambda x, pos: str(int(x)))
            if not sharex or sharex and ix + 1 == len(seles):
                ax.set_xticks(arange, labels=labels, rotation=90)
                ax.locator_params(axis="x", tight=True, nbins=nbins)
                for label in ax.xaxis.get_majorticklabels():
                    label.set_verticalalignment("top")
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
        assert len(fpts) > 1, "HCA requires multiple fingerprints, please add more selections."
        
        dendro, medoids = plot_hca_base(
            corrs,
            labels,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            hide_threshold=hide_threshold,
            annotate=annotate,
            axis=ax,
        )
        for label in ax.xaxis.get_majorticklabels():
            label.set_horizontalalignment("right")
        if isinstance(plot_hca, (str, Path)):
            fig.savefig(plot_hca, dpi=300)
            if not quiet:
                print(f"HCA figure saved to {plot_hca}")
    
    return fpts, corrs, dendro, medoids


@declare_command
def get_ho(
    hs1: Selection,
    hs2: Selection,
    radius: float = 2.5,
    quiet: int = True,
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
    dist = distance_matrix(atoms1, atoms2) - radius <= 0
    num_contacts1 = np.sum(np.any(dist, axis=1))
    num_contacts2 = np.sum(np.any(dist, axis=0))
    ho = (num_contacts1 + num_contacts2) / (len(atoms1) + len(atoms2))
    if not quiet:
        print(f"HO: {ho:.2f}")
    return ho


class ResidueSimilarityMethod(StrEnum):
    JACCARD = "jaccard"
    OVERLAP = "overlap"


@declare_command
def res_sim(
    hs1: Selection,
    hs2: Selection,
    radius: float = 4.0,
    seq_align: bool = False,
    method: ResidueSimilarityMethod = ResidueSimilarityMethod.JACCARD,
    quiet: int = True,
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
    group1 = pm.get_property("Group", hs1)
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


class SimilarityFunc(StrEnum):
    HO = "ho"
    RESIDUE_JACCARD = "residue_jaccard"
    RESIDUE_OVERLAP = "residue_overlap"


@declare_command
def plot_pairwise_hca(
    sele: Selection,
    function: SimilarityFunc = SimilarityFunc.HO,
    radius: float = 2.0,
    align: bool = False,
    annotate: bool = False,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: float = 0.0,
    hide_threshold: bool = False,
    plot: str = "",
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
                case SimilarityFunc.HO:
                    ret = get_ho(obj1, obj2, radius=radius)
                case SimilarityFunc.RESIDUE_JACCARD:
                    ret = res_sim(
                        obj1,
                        obj2,
                        radius=radius,
                        method=ResidueSimilarityMethod.JACCARD,
                        seq_align=align,
                    )
                case SimilarityFunc.RESIDUE_OVERLAP:
                    ret = res_sim(
                        obj1,
                        obj2,
                        radius=radius,
                        method=ResidueSimilarityMethod.OVERLAP,
                        seq_align=align,
                    )
            X.append(1 - ret)
    dendro, medoids = plot_hca_base(X, objects, linkage_method, color_threshold, hide_threshold, annotate, plot)
    return X, objects, dendro, medoids


class PrioritizationType(StrEnum):
    RESIDUE = "residue"
    ATOM = "atom"


@declare_command
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


@declare_command
def plot_euclidean_hca(
    exprs: Selection,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    color_threshold: float = 0.0,
    hide_threshold: bool = False,
    annotate: bool = False,
    plot: str = None,
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
    assert len(set(pm.get_property("Type", o) for o in object_list)) == 1, "Only hotspots of the same type are allowed in the euclidean HCA."

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
        labels.append(obj)
        x, y, z = np.mean(pm.get_coordset(obj), axis=0)
        if hs_type == "K15":
            ST = pm.get_property("ST", obj)
            S0 = pm.get_property("S0", obj)
            CD = pm.get_property("CD", obj)
            MD = pm.get_property("MD", obj)
            p[idx, :] = np.array([ST, S0, CD, MD, x, y, z])
        elif hs_type == "CS":
            ST = pm.get_property("ST", obj)
            p[idx, :] = np.array([ST, x, y, z])
        elif hs_type == "ACS":
            ST = pm.get_property("ST", obj)
            MD = pm.get_property("MD", obj)
            p[idx, :] = np.array([ST, MD, x, y, z])
        
    X = []
    for idx1, obj1 in enumerate(object_list):
        for idx2, obj2 in enumerate(object_list):
            if idx1 >= idx2:
                continue
            p1 = p[idx1, :]
            p2 = p[idx2, :]
            d = np.sqrt(np.sum((p1-p2)**2))
            X.append(d)
    X = np.array(X)
    X = (X-X.min())/(X.max()-X.min())
    return plot_hca_base(X, labels, linkage_method, color_threshold, hide_threshold, annotate, plot)


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
        self.maxLengthSpin.setValue(3)
        self.maxLengthSpin.setMinimum(3)
        self.maxLengthSpin.setMaximum(8)
        boxLayout.addRow("Max length:", self.maxLengthSpin)

        self.runFpocketCheck = QCheckBox()
        self.runFpocketCheck.setChecked(False)
        boxLayout.addRow("Calc Fpocket, DL/BL and Egbert (2019):", self.runFpocketCheck)

        self.bekarLabel = QLineEdit()
        boxLayout.addWidget(self.bekarLabel)
        boxLayout.addRow("Bekar-Cesaretli (2025) label:", self.bekarLabel)

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
        self.bekarLabel.setText("")

    def load(self):
        run_fpocket = self.runFpocketCheck.isChecked()
        max_length = self.maxLengthSpin.value()
        bekar_label = self.bekarLabel.text().strip()
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
            k15_max_length=max_length,
            run_fpocket=run_fpocket,
            bekar_label=bekar_label
        )


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
        self.current_tab = "K15"

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
            ("Kozakov2015", "K15"): [
                "Class",
                "ST",
                "S0",
                "CD",
                "MD",
                "Length",
                "Fpocket",
            ],
            ("CS", "CS"): ["ST"],
            ("ACS", "ACS"): ["Class", "ST", "MD"],
            ("Bekar-Cesaretli2025", "BC25"): ["Total", "CS16", "K15D", "IsBekar"],
            ("Egbert2019", "E19"): ["Fpocket", "ST", "S0", "S1", "S2", "S3", "Length"],
            ("Fpocket", "Fpocket"): ["Pocket Score", "Drug Score"],
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
        if self.current_tab == "BC25":
            self.selected_objs = pm.get_object_list(sele)
        else:
            self.selected_objs = pm.get_object_list(
                f"({sele}) & (*.{self.current_tab}*)",
            )
        if self.selected_objs is None:
            self.selected_objs = set()
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


class SimilarityWidget(QWidget):

    def __init__(self):
        super().__init__()

        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)

        groupBox = QGroupBox("General")
        mainLayout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.hotspotSeleLine = QLineEdit()
        boxLayout.addRow("Hotspots:", self.hotspotSeleLine)

        self.annotateCheck = QCheckBox()
        self.annotateCheck.setChecked(True)
        boxLayout.addRow("Annotate:", self.annotateCheck)

        self.linkageMethodCombo = QComboBox()
        self.linkageMethodCombo.addItems([e.value for e in LinkageMethod])
        boxLayout.addRow("Linkage:", self.linkageMethodCombo)

        self.colorThresholdSpin = QDoubleSpinBox()
        self.colorThresholdSpin.setMinimum(0)
        self.colorThresholdSpin.setMaximum(10)
        self.colorThresholdSpin.setValue(0)
        self.colorThresholdSpin.setSingleStep(0.1)
        self.colorThresholdSpin.setDecimals(2)
        boxLayout.addRow("Color threshold:", self.colorThresholdSpin)

        self.hideThresholdCheck = QCheckBox()
        self.hideThresholdCheck.setChecked(False)
        boxLayout.addRow("Hide threshold:", self.hideThresholdCheck)

        layout = QHBoxLayout()
        mainLayout.addLayout(layout)

        groupBox = QGroupBox("Pairwise similarity")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.functionCombo = QComboBox()
        self.functionCombo.addItems([e.value for e in SimilarityFunc])
        boxLayout.addRow("Function:", self.functionCombo)

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
        plotButton.clicked.connect(self.plot_pairwise)
        boxLayout.addWidget(plotButton)

        groupBox = QGroupBox("N-dimensional euclidean")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_euclidean_hca)
        boxLayout.addWidget(plotButton)

    def plot_pairwise(self):
        sele = self.hotspotSeleLine.text()
        function = self.functionCombo.currentText()
        radius = self.radiusSpin.value()
        align = self.pairwiseSeqAlignCheck.isChecked()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        hide_threshold = self.hideThresholdCheck.isChecked()
        annotate = self.annotateCheck.isChecked()

        plot_pairwise_hca(
            sele,
            function,
            radius,
            align,
            annotate,
            linkage_method,
            color_threshold,
            hide_threshold,
        )

    def plot_euclidean_hca(self):
        sele = self.hotspotSeleLine.text()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        hide_threshold = self.hideThresholdCheck.isChecked()
        annotate = self.annotateCheck.isChecked()

        return plot_euclidean_hca(
            sele,
            linkage_method,
            color_threshold,
            hide_threshold,
            annotate,
        )


class CountWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        self.setLayout(layout)

        groupBox = QGroupBox("Color projection")
        layout.addWidget(groupBox)
        boxLayout = QFormLayout()
        groupBox.setLayout(boxLayout)

        self.multiSeleLine = QLineEdit()
        boxLayout.addRow("Expressions:", self.multiSeleLine)

        self.proteinExpressionLine = QLineEdit()
        boxLayout.addRow("Protein:", self.proteinExpressionLine)

        self.radiusSpin = QDoubleSpinBox()
        self.radiusSpin.setValue(3)
        self.radiusSpin.setDecimals(2)
        self.radiusSpin.setSingleStep(0.5)
        self.radiusSpin.setMinimum(2)
        self.radiusSpin.setMaximum(10)
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

        self.multiSelesLine = QLineEdit("")
        boxLayout.addRow("Multi sele:", self.multiSelesLine)

        self.siteSelectionLine = QLineEdit("*")
        boxLayout.addRow("Focus site:", self.siteSelectionLine)

        self.radiusSpin = QDoubleSpinBox()
        self.radiusSpin.setValue(3)
        self.radiusSpin.setDecimals(2)
        self.radiusSpin.setSingleStep(0.5)
        self.radiusSpin.setMinimum(2)
        self.radiusSpin.setMaximum(10)
        boxLayout.addRow("Radius:", self.radiusSpin)

        self.fingerprintsCheck = QCheckBox()
        self.fingerprintsCheck.setChecked(False)
        boxLayout.addRow("Fingerprints:", self.fingerprintsCheck)
        @self.fingerprintsCheck.stateChanged.connect
        def stateChanged(checkState):
            if checkState == QtCore.Qt.Checked:
                fptBox.setEnabled(True)
            else:
                fptBox.setEnabled(False)

        fptBox = QGroupBox()
        boxLayout.addRow(fptBox)
        boxLayout.setWidget(boxLayout.rowCount(), QFormLayout.SpanningRole, fptBox)
        fptLayout = QFormLayout()
        fptBox.setLayout(fptLayout)
        fptBox.setEnabled(False)

        self.nBinsSpin = QSpinBox()
        self.nBinsSpin.setValue(20)
        self.nBinsSpin.setMinimum(0)
        self.nBinsSpin.setMaximum(500)
        fptLayout.addRow("Num bins:", self.nBinsSpin)

        self.sharexCheck = QCheckBox()
        self.sharexCheck.setChecked(True)
        fptLayout.addRow("Share x axis:", self.sharexCheck)

        self.hcaCheck = QCheckBox()
        self.hcaCheck.setChecked(False)

        boxLayout.addRow("HCA:", self.hcaCheck)
        @self.hcaCheck.stateChanged.connect
        def stateChanged(checkState):
            if checkState == QtCore.Qt.Checked:
                hcaBox.setEnabled(True)
            else:
                hcaBox.setEnabled(False)

        hcaBox = QGroupBox()
        boxLayout.addRow(hcaBox)
        boxLayout.setWidget(boxLayout.rowCount(), QFormLayout.SpanningRole, hcaBox)
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

        self.hideThresholdCheck = QCheckBox()
        self.hideThresholdCheck.setChecked(False)
        hcaLayout.addRow("Hide threshold:", self.hideThresholdCheck)
        
        plotButton = QPushButton("Plot")
        plotButton.clicked.connect(self.plot_fingerprint)
        boxLayout.addWidget(plotButton)

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
        radius = self.radiusSpin.value()
        plot_fingerprints = self.fingerprintsCheck.isChecked()
        plot_hca = self.hcaCheck.isChecked()
        nbins = self.nBinsSpin.value()
        sharex = self.sharexCheck.isChecked()
        annotate = self.annotateCheck.isChecked()
        linkage_method = self.linkageMethodCombo.currentText()
        color_threshold = self.colorThresholdSpin.value()
        hide_threshold = self.hideThresholdCheck.isChecked()

        fpt_sim(
            multi_seles,
            site,
            radius,
            plot_fingerprints=plot_fingerprints,
            plot_hca=plot_hca,
            nbins=nbins,
            sharex=sharex,
            annotate=annotate,
            linkage_method=linkage_method,
            color_threshold=color_threshold,
            hide_threshold=hide_threshold,
        )
        plt.show()


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(600, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("(XDrugPy) Hotspots")

        tab = QTabWidget()
        tab.addTab(LoadWidget(), "Load")
        tab.addTab(TableWidget(), "Properties")
        tab.addTab(SimilarityWidget(), "HCA")
        tab.addTab(CountWidget(), "Fingerprinting")

        layout.addWidget(tab)


dialog = None


def run_plugin_gui():
    global dialog
    if dialog is None:
        dialog = MainDialog()
    dialog.show()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("(XDrugPy) Hotspots", run_plugin_gui)
