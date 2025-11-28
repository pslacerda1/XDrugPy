##############################################
# HCA of a large number of hotspots
#
# Runs like:
#    $ pymol -c hca.py
#
#############################################

from os.path import expanduser, expandvars
from glob import glob
from xdrugpy.hotspots import load_ftmap, plot_pairwise_hca
from pymol import cmd as pm
from matplotlib import pyplot as plt


# OPTIONAL: Optimization that disables an unuseful feature if we'll not
# launch the graphical interface after script execution.
pm.undo_disable()

# List the FTMap PDB files but limits arbitrarily to the first 25 entries.
# This glob ???? match four chars (like a PDB id), but could be replaced by:
#                                   ~/Desktop/PEPTI/atlas/*_atlas.pdb
files = glob(expandvars(expanduser("~/Desktop/PEPTI/atlas/????_atlas.pdb")))
for file in files[:25]:
    
    # Load structures and does hotspots ligability analysis.
    ftmap = load_ftmap(file)

    # OPTIONAL: After each read, exclude hotspots that doesn't satisfy the
    # plotting criteria. It also preserves the *.protein structures from
    # exclusion. This optimization reduces the number of PyMOL objects
    # irrelevant to our analysis and improves the performance at high loads.
    for obj in pm.get_object_list("NOT ((*.K15_* AND p.S0>25) OR *.protein)"):
        pm.delete(obj)

# Does the actual hierarchical cluster analysis (HCA) with hotspot overlap (HO)
# function between remaining hotspots after loads and deletions of objects.
plot_pairwise_hca(
    '*.K15_* AND p.S0>25',  # all groups, only Kozakov2015, any class
    align=False,            # suposes previously aligned structures (FTMove?)
    radius=4,
    color_threshold=0.7,    # probably you'll need to adjust this variable
    hide_threshold=True,
    annotate=False
)
plt.show()  # displays the HCA