##########################################
# HCA of maybe a large number of hotspots
#
# Run it like:
#    $ pymol -c hca.py
##########################################

from os.path import expanduser, expandvars
from glob import glob
from xdrugpy.hotspots import load_ftmap, plot_pairwise_hca, plot_euclidean_hca, SimilarityFunc, LinkageMethod

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
    #                             "NOT (*.K15_* AND p.S0>20 OR *.protein)"
    for obj in pm.get_object_list("NOT (*.K15_* AND p.S0>20)"):
        pm.delete(obj)

# If you don't delete protein structures you can have a nice session!
pm.save("~/My Folder/nice_session.pze")

# Does the actual hierarchical cluster analysis (HCA) with hotspot overlap (HO)
# function. Only strong Kozakov2015 hotspots with at least p.S0>20 that remained
# from the previous successive loading and deletions.
plot_pairwise_hca(
    '*.K15_* AND p.S0>20',
    function=SimilarityFunc.HO,
    align=False,                # suposes previously aligned structures (FTMove?)
    radius=1.5,
    color_threshold=0,          # probably you'll need to adjust this variable
    hide_threshold=False,
    annotate=True
)
plt.show()  # displays the HCA

# A more standard type of HCA over the same hotspots of the previous analysis.
# However this ones calculates the distance over the aggregation of all hotspot
# properties, including coordinates of center-of-mass.
plot_euclidean_hca(
    '*.K15_* AND p.S0>20',
    linkage_method=LinkageMethod.WARD,
    color_threshold=0,
    hide_threshold=False,
    annotate=True,
)
plt.show()  # displays the HCA
