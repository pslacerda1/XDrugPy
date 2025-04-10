from pymol import cmd as pm
import numpy as np
import pandas as pd
from fnmatch import fnmatchcase
from collections import namedtuple
from functools import lru_cache

from .utils import declare_command, Selection


Residue = namedtuple("Reisude", "model index resn resi chain x y z")

lru_cache(999999999)
def get_residue_from_object(obj, idx):
    res = []
    pm.iterate_state(
        -1,
        f"%{obj} & index {idx}",
        'res.append(Residue(model, int(index), resn, int(resi), chain, float(x), float(y), float(z)))',
        space={'res': res, 'Residue': Residue}
    )
    return res[0]


@declare_command
def get_mapping(
    ref_polymer: Selection,
    other_polymers: Selection,
    site: str = '*',
    radius: float = 2,
):    
    # Get polymers to be mapped to reference site
    polymers = set()
    for obj in pm.get_object_list("polymer"):
        if fnmatchcase(obj, other_polymers):
            polymers.add(obj)
    if ref_polymer in polymers:
        polymers.remove(ref_polymer)

    # Do the alignmnet
    mappings = np.empty((0, 8))
    for polymer in polymers:
        try:
            aln_obj = pm.get_unused_name()
            pm.cealign(
                ref_polymer, polymer, transform=0, object=aln_obj
            )
            aln = pm.get_raw_alignment(aln_obj)
        finally:
            pm.delete(aln_obj)
        for (obj1, idx1), (obj2, idx2) in aln:
            res1 = get_residue_from_object(obj1, idx1)
            res2 = get_residue_from_object(obj2, idx2)
            mappings = np.vstack([mappings, res1, res2])
    return pd.DataFrame(mappings, columns=Residue._fields)