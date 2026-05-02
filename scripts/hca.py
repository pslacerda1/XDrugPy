#
# Hierarchical Cluster Analysis (HCA) of a large number of hotspots
#
from os.path import expanduser
from glob import glob
from xdrugpy.hotspots import (
    load_ftmap,
    plot_multivariate_hca,
    LinkageMethod
)
from pymol import cmd as pm
from matplotlib import pyplot as plt
import matplotlib.style
import matplotlib.colors
from cycler import cycler

# Change the default Matplotlib style and parameters to improve
# the aesthetics of generated figures.
matplotlib.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 6),
    'svg.fonttype': 'none',
    'axes.prop_cycle': cycler(color=reversed(
        matplotlib.colors.XKCD_COLORS
    )),
})

# OPTIONAL: Optimization that disables an unuseful feature if we'll not
# launch the graphical interface after script execution.
pm.undo_disable()

# Adapt the string to list all your FTMove PDB files but limits arbitrarily to
# the first 5 entries found. Use small lists when developing the script to speed
# up the process
files = glob(expanduser("~/Desktop/PEPTI/atlas/*_atlas.pdb"))
for file in files[:5]:
    
    # Load structures and does hotspots ligability analysis.
    ftmap = load_ftmap(file)

    # OPTIONAL: After each load, delete hotspots that doesn't satisfy the
    # plotting criteria. It also deletes *.protein structures. This
    # optimization reduces the number of PyMOL objects irrelevant to our
    # analysis improving the performance at high loads.
    for obj in pm.get_object_list("NOT (*.D* AND p.S0>20)"):
    #                             "NOT (*.D* AND p.S0>20 OR *.protein)"
        pm.delete(obj)

# If you don't delete protein structures you can have a nice session!
# pm.save("~/My Folder/nice_session.pze")

# The standard HCA over the same hotspots of the previous analysis. This one
# calculates the distance over the aggregation of hotspot properties, including
# coordinates of center-of-mass.
plot_multivariate_hca(
    '*.D* AND p.S0>20',
    linkage_method=LinkageMethod.WARD,
    color_threshold=2,
    only_medoids=True,
    annotate=False,
)
plt.title("Multivariate HCA")

# Comment/Uncomment one of the following lines to show or save the figure.
#plt.show()  # good for local usage, but disable for headless or remote sessions
#plt.savefig("multivariate_hca.svg")  # save the SVG or PNG file
