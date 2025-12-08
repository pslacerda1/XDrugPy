#
# Fingerprint based HCA 
#
from os.path import expanduser, basename, splitext
from glob import glob

from pymol import cmd as pm
from matplotlib import pyplot as plt

from xdrugpy.hotspots import (
    load_ftmap,
    fpt_sim,
    LinkageMethod
)

pm.undo_disable()
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 6)
})

# limit the number of files for testing or use all=999
files = glob(expanduser("~/Desktop/Qualificação/FTMove_1JQH_A/FTMap/????_?.pdb"))
groups = []
for file in files[:9999]:

    # Save FTMap group for later use.
    group = basename(splitext(file)[0])
    groups.append(group)

    # Load structures and does hotspots ligability analysis.
    # Group results togheter by group name.
    ftmap = load_ftmap(file, group)

    # After each load, delete objects doesn't used in plotting.
    for obj in pm.get_object_list(f"NOT (*.protein OR *.CS_*)"):
        pm.delete(obj)

fpt_sim(
    ' / '.join(f'{g}.CS_*' for g in groups),
    site='*',
    radius=3.0,
    plot_fingerprints=False, # too much structures for individual fingerprint plots
    plot_hca=True,
    linkage_method=LinkageMethod.WARD,
    color_threshold=1.5,    # limit to color clusters with distance < 2.0
    hide_threshold=True,    # hide every hotspot below 2.0 except medoids?
    annotate=False,         # is desirable the value at each cell?
    quiet=True,             # print individual correlation values?
)
plt.show()  # display plot
