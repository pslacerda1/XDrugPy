#
# Hierarchical Cluster Analysis (HCA) of a large number of hotspots
#
from os.path import expanduser
from glob import glob
from xdrugpy import (
    load_ftmap,
    calc_multivariate_hca,
    LinkageMethod,
    configure_matplotlib,
    plot,
)
from pymol import cmd as pm

# Change the default Matplotlib style and parameters to improve
# the aesthetics of generated figures.
configure_matplotlib(
    style='default',
    params={
        'font.size': 14,
        'figure.figsize': (10, 6)
    }
)

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
# pm.save("~/New Folders/almost_nice_session.pze")

# The standard HCA based on the distance between hotspot properties, including
# coordinates of center-of-mass.
calc_multivariate_hca(
    '*.D* AND p.S0>20',
    linkage_method=LinkageMethod.WARD,
    color_threshold=2,
    only_medoids=True,
    annotate=False,
)
plot() # or plot("~/MyFolder/hca.png") to save the figure instead of showing it