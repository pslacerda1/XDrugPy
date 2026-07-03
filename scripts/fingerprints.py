from glob import glob
from matplotlib import pyplot as plt
from pymol import cmd as pm

from xdrugpy import (
    load_ftmap,
    calc_fingerprints,
    LinkageMethod,
    configure_matplotlib,
    plot
)

# my opinion
configure_matplotlib('default', {   # the style
    'font.size': 14,                # small font
    'figure.figsize': (10, 8),      # good dimensions
    'svg.fonttype': 'none'          # editable with notepad
})


# Load FTMap files
for filename in glob("/home/peu/Desktop/Qualificação/TR_Dímero/test/*.pdb"):
    load_ftmap(filename)

# Delete low strength hotspots, it's just noise!
for obj in pm.get_object_list('*.CS.* AND p.ST<5'):
    pm.delete(obj)

# Run the fingerprint similarity analisys
calc_fingerprints(
    # This long multi-line string has the desired
    # hotspots separated by slashes.
    multi_seles="""
        1BZL.CS.* / 2JK6.CS.* / 2TPR.CS.* /
        2W0H.CS.* / 2WPF.CS.* / 6BU7.CS.*
    """,
    # Focus the analysis to a site. Because the
    # first object is 1BZL, the site can be given
    # by a ligand or residue index of that 
    # structure also.
        #site='resi 207-209+187-189',
        # site='*',
    site='resi 436',
    # How many angstroms will be given around the
    # site? It can be 0 (exclusively resi 436),
    # 5 (a large padding), or any other value.
    site_radius=5,
    # Apply multiple sequence alignment
    seq_align_omega=True,
    # How far from residues to look for probes.
    contact_radius=4.0,
    # Do plot fingerprints.
    fingerprints_axis=True,
    # Fingerprints will have 40 labels.
    nbins=40,
    # Fingerprints will share the the x axis and
    # display residues from the first mentionated
    # conformation (1JNW).
    sharex=True,
    # Do plot the HCA.
    plot_hca=True,
    # Annotate the HCA heatmap with cell values.
    annotate=True,
    # Do not cluster the HCA by height (color)
    # threshold.
    color_threshold=0.0,
    # Ward is cool.
    linkage_method=LinkageMethod.WARD,
)
plot()
plt.waitforbuttonpress()
