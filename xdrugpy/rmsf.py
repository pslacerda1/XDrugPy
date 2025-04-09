from pymol import cmd as pm
from matplotlib import pyplot as plt
from xdrugpy.utils import ONE_LETTER
from collections import defaultdict
from functools import lru_cache
import numpy as np
from fnmatch import fnmatch


def rmsf(ref_site, prot_expr, site_margin=5, ax=None):
    frames = []
    for obj in pm.get_object_list():
        if fnmatch(obj, prot_expr):
            frames.append(obj)

    f0 = frames[0]
    site = set()
    pm.iterate(
        f'(%{f0} & polymer) within {site_margin} of ({ref_site})',
        'site.add((resn,resi,chain))',
        space={'site': site}
    )
    @lru_cache(25000)
    def get_residues(frame, qualifier="name CA"):
        residues = []
        coords = np.empty((0, 3))
        chains = pm.get_chains(f"{frame} & polymer")
        sele = f"{qualifier} "
        for chain in chains:
            sele += f" | (c. {chain} &"
            idx_resids = "+".join(r[1] for r in site if r[2] == chain)
            sele += f' i. {idx_resids}'
            sele += ") "
        for a in pm.get_model(sele).atom:
            resi = (a.resn, a.resi, a.chain)
            if resi in site:
                residues.append(resi)
                coords = np.vstack([coords, a.coord])
        return residues, coords

    mean_pos = defaultdict(list)
    for fr in frames:
        resis, coordinates = get_residues(fr)
        for resi, coords in zip(resis, coordinates):
            mean_pos[resi].append(coords)
    for resi in set(resis):
        mean_pos[resi] = np.mean(mean_pos[resi], axis=0)

    # Aggregate coords from all frames
    X = {}
    for fr in frames:
        resis, coordinates = get_residues(fr)
        for resi, coords in zip(resis, coordinates):
            if resi not in X:
                X[resi] = np.empty((0,3))
            X[resi] = np.vstack([X[resi], coords])
            
    # Find mean positions for each reisude
    mean_positions = defaultdict(list)
    for fr in frames:
        resis, coordinates = get_residues(fr)
        for resi, coords in zip(resis, coordinates):
            mean_positions[resi].append(coords)
    for resi in set(resis):
        mean_positions[resi] = np.mean(mean_positions[resi], axis=0)

    # Sort residues
    X = {k: X[k] for k in sorted(X, key=lambda z: (z[2], z[1]))}

    # Calculate RMSF
    RMSF = []
    LABELS = []
    for resi, coords in X.items():
        rmsf = np.sum((coords - mean_positions[resi]) ** 2) / coords.shape[0]
        rmsf = np.sqrt(rmsf)
        label = '%s %s:%s' % (ONE_LETTER[resi[0]], resi[1], resi[2])
        LABELS.append(label)
        RMSF.append(rmsf)

    ax_file = False
    if ax is None:
        fig, ax = plt.subplots()
    elif isinstance(ax, str):
        ax_file = ax
        fig, ax = plt.subplots()

    ax.bar(LABELS, RMSF)
    ax.set_xlabel("Residue")
    ax.set_ylabel("RMSF (Å)")
    ax.tick_params(axis='x', rotation=90)

    if ax_file:
        plt.tight_layout()
        plt.savefig(ax_file)
    
    return RMSF, LABELS