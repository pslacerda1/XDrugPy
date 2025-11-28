##############################################
# HCA of a large number of hotspots
# Runs like:
#    $ pymol -c hca.py
#

from os.path import expanduser, expandvars
from glob import glob
from xdrugpy.hotspots import load_ftmap, plot_pairwise_hca
from pymol import cmd as pm
from matplotlib import pyplot as plt


# OPTIONAL: Optimization that disables a unuseful feature if we'll not
# launch the graphical interface after script execution.
pm.undo_disable()

# Reads the FTMap structures but limits arbitrarily to the first 25 entries.
files = expandvars(expanduser("~/Desktop/PEPTI/atlas/????_atlas.pdb"))
for file in glob(files)[:25]:
    
    # Read files and analysis hotspots modulability.
    ftmap = load_ftmap(file)

    # OPTIONAL: At each read, exclude hotspots that doesn't satisfy the plotting
    # criteria. It also preserves the *.protein structures from exclusion. It is
    # an optimization that reduce the number of PyMOL objects irrelevant to our
    # pretentions and improving performance at high loads.
    for obj in pm.get_object_list("NOT (*.K15_* AND p.S0>25 OR *.protein)"):
        pm.delete(obj)

# Does the actual hierarchical cluster analysis (HCA) with hotspot overlap (HO)
# data between remaining hotspots after loads and deletions of hotspot objects.
plot_pairwise_hca(
    '*.K15_*',          # all groups, only Kozakov2015, any class
    align=False,        # suposes previously aligned structures (FTMove?)
    radius=4,
    color_threshold=0.7, # probably you'll need to adjust this variable
    hide_threshold=True,
    annotate=False
)
plt.show()  # displays the HCA