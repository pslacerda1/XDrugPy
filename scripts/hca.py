##############################################
# HCA of a large number of hotspots
#
# Runs like:
#    $ pymol -c hca.py
#
#############################################

from os.path import expanduser, expandvars
from glob import glob
from xdrugpy.hotspots import load_ftmap, plot_pairwise_hca, SimilarityFunc

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

    # OPTIONAL: After each load, delete hotspots that doesn't satisfy the
    # plotting criteria. It also deletes the *.protein structures. This
    # optimization reduces the number of PyMOL objects irrelevant to our
    # analysis improving the performance at high loads.
    for obj in pm.get_object_list("NOT (*.K15_* AND p.S0>25)"):
        pm.delete(obj)

# Does the actual hierarchical cluster analysis (HCA) with hotspot overlap (HO)
# function. Only very strong Kozakov2015 hotspots with at least p.S0>25 that
# remained from the previous successive loading and deletions.
plot_pairwise_hca(
    '*.K15_* AND p.S0>25',      # all groups, K15 & S0>25, of any class
    function=SimilarityFunc.HO, # superposition based hotspot similarity method
    align=False,                # suposes previously aligned structures (FTMove?)
    radius=4,
    color_threshold=0.7,        # probably you'll need to adjust this variable
    hide_threshold=True,
    annotate=False
)
plt.show()  # displays the HCA