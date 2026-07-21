#
# Hierarchical Cluster Analysis.
#
# You may need this script to process a large number of FTMap
# files (like from FTMove or Atlas) without consuming too much
# computer RAM (memory) like in the graphical user interface.
#
from glob import glob
from matplotlib import pyplot as plt
from pymol import cmd as pm
from xdrugpy import (
    load_ftmap,
    calc_multivariate_hca,
    calc_univariate_hca,
    LinkageMethod,
    MultivariateDistanceMethod,
    UnivariateDistanceMethod,
    configure_matplotlib,
)

# Change the default Matplotlib style and parameters to improve
# the aesthetics of generated figures to your suits. Let's tweak!
configure_matplotlib(
    style='default', # the style theme
    params={
        'font.size': 10,            # small font for the dimensions
        'figure.figsize': (10, 8),  # good dimensions
        'svg.fonttype': 'none'      # editable with notepad if SVG
    }
)

# List all your FTMove PDB files but limits to the first 5
# entries found. Use small lists when developing the script
# to speed up the process.
files = glob("/home/peu/Desktop/PEPTI/atlas/*_atlas.pdb")
for file in files[:5]:
    
    # Load structures and does hotspots ligability analysis.
    ftmap = load_ftmap(
        file,
        deep_search=True,     # enable full combinatory search
        remove_nested=True,   # remove hotspots fully inside others
    )

    # OPTIONAL: After each load, delete hotspots that doesn't satisfy the
    # plotting criteria. It also deletes *.protein structures. This
    # optimization reduces the number of PyMOL objects irrelevant to our
    # analysis improving the performance at high loads.
    for obj in pm.get_object_list("NOT (*.D* AND p.S0>20)"):
    #                             "NOT (*.D* AND p.S0>20 OR *.protein)"
        pm.delete(obj)

# If you don't delete protein structures you can have a nice session!
# pm.save("/home/peu/MyFolder/almost_nice_session.pze")

# The standard HCA based on the distance between hotspot properties, including
# coordinates of center-of-mass.
calc_multivariate_hca(
    '*.D* AND p.S0>20',
    dist_method=MultivariateDistanceMethod.EUCLIDEAN,
    linkage_method=LinkageMethod.WARD,
    nclusters=5,
    only_medoids=True,
    annotate=False,
    figure_title="MyProj HCA (Multivariate)",
    dendrogram_plot="/home/peu/Downloads/dendro.png",  # you can use png
    heatmap_plot="/home/peu/Downloads/heat.svg",       # or other formats
)

# Dendrogram cluster analysis with distances based on reciprocal fractional
# overlap average between hotspot objects.
calc_univariate_hca(
    '*.D* AND p.S0>20',
    dist_methdo=UnivariateDistanceMethod.FO_AVG,
    linkage_method=LinkageMethod.AVERAGE,
    color_threshold=0.35,
    annotate=False,
    figure_title="MyProj HCA (Univariate)",
    dendrogram_plot=True,   # can also opt to show on the screen
    heatmap_plot=True,      # and save the image from the pop up window
                            # or even set to False
)
plt.waitforbuttonpress()
