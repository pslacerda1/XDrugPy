from pymol import cmd as pm
from collections import defaultdict
from functools import lru_cache
import numpy as np
from fnmatch import fnmatch

from .utils import ONE_LETTER, declare_command, mpl_axis, plot_hca_base, multiple_expression_selector


@declare_command
def rmsf(
    ref_site: str,
    prot_expr: str,
    site_margin:float = 3.0,
    qualifier: str = 'name CA',
    pretty: bool = True,
    axis: str = ''
):
    """
    DESCRIPTION
        Calculate the RMSF of muliple related structures.

        A reference site must be supplied to focus, however full protein
        analysis can be achieved with a star * wildcard. A expression
        based on fnmatch select the structures to calculate the RMSF.
    """
    frames = []
    for obj in pm.get_object_list():
        if fnmatch(obj, prot_expr):
            frames.append(obj)

    f0 = frames[0]
    site = set()
    pm.iterate(
        f'(%{f0} & polymer) within {site_margin} of ({ref_site})',
        'site.add((resn,int(resi),chain))',
        space={'site': site}
    )
    
    @lru_cache(25000)
    def get_residues(frame):
        residues = []
        coords = np.empty((0, 3))
        chains = pm.get_chains(f"{frame} & polymer")
        sele = f"{frame} & {qualifier} & ("

        for i, chain in enumerate(chains):
            if i == 0:
                sele += f"(c. {chain} &"
            else:
                sele += f" | (c. {chain} &"
            idx_resids = "+".join(str(r[1]) for r in site if r[2] == chain)
            sele += f' i. {idx_resids}'
            sele += ") "
        sele += ")"
        
        for a in pm.get_model(sele).atom:
            resi = (a.resn, int(a.resi), a.chain)
            if resi in site:
                residues.append(resi)
                coords = np.vstack([coords, a.coord])
        return residues, coords

    # Aggregate coords from all frames
    X = {}
    for fr in frames:
        resis, coordinates = get_residues(fr)
        for resi, coords in zip(resis, coordinates):
            if resi not in X:
                X[resi] = np.empty((0,3))
            X[resi] = np.vstack([X[resi], coords])

    # Sort residues
    X = {k: X[k] for k in sorted(X, key=lambda z: (z[2], z[1]))}

    # Find mean positions for each reisude
    means = {}
    for resi in X:
        means[resi] = np.mean(X[resi], axis=0)

    # Calculate RMSF
    RMSF = []
    LABELS = []
    pm.alter(f"{f0} & polymer", "p.rmsf=0.0")
    for resi, coords in X.items():
        rmsf = np.sum((coords - means[resi]) ** 2) / coords.shape[0]
        rmsf = np.sqrt(rmsf)
        label = '%s %s:%s' % (ONE_LETTER[resi[0]], resi[1], resi[2])
        pm.alter(f"{f0} & i. {resi[1]} & c. {resi[2]}", f"p.rmsf={rmsf}")
        LABELS.append(label)
        RMSF.append(rmsf)
    
    # Show data
    if pretty:
        pm.hide('everything', f"{f0} & polymer")
        pm.show_as("line", f"{f0} & polymer")
        pm.spectrum("p.rmsf", "rainbow", f"{f0} & polymer")

    with mpl_axis(axis) as ax:
        ax.bar(LABELS, RMSF)
        ax.set_xlabel("Residue")
        ax.set_ylabel("RMSF")
        ax.tick_params(axis='x', rotation=90)
    
    return RMSF, LABELS


@declare_command
def rmsd_hca(
    ref_site: str,
    prot_expr: str,
    qualifier: str = 'name CA',
    site_margin: float = 5.0,
    linkage_method: str = 'ward',
    color_threshold: float = 0.0,
    axis: str = ''
):
    """
    DESCRIPTION
        Calculate the RMSD of multiple related structures. First it realizes
        multiple sequence/structure alignment with the cealign function in
        order to get the equivalent atoms, so it can be realized between
        relatively distant homologues.

        A reference site must be supplied to focus, however full protein
        analysis can be achieved with a star * wildcard. A expression
        based on fnmatch select the structures to calculate the RMSD.
    """
    frames = []
    for obj in pm.get_object_list():
        for expr in prot_expr.split():
            if fnmatch(obj, expr):
                frames.append(obj)
    f0 = frames[0]

    site = set()
    pm.iterate(
        f'(%{f0} & polymer) within {site_margin} of ({ref_site})',
        'site.add((resn,resi,chain))',
        space={'site': site}
    )
    
    # Aggregate coords from all frames
    X = []
    for i1, f1 in enumerate(frames):
        for i2, f2 in enumerate(frames):
            if i1 >= i2:
                continue
            rmsd = pm.rms(
                f"(%{f1} & polymer & {qualifier}) within {site_margin} of ({ref_site})",
                f"(%{f2} & polymer & {qualifier}) within {site_margin} of ({ref_site})",
            )
            X.append(rmsd)
    return plot_hca_base(X, frames, linkage_method, color_threshold, axis)

def __init_plugin__(app=None):
    pass