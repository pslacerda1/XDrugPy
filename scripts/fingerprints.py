from glob import glob
from pymol import cmd as pm
from matplotlib import pyplot as plt
from xdrugpy import (
    load_ftmap,
    calc_fingerprints,
    LinkageMethod,
    configure_matplotlib,
)

# tweak this for you suits
configure_matplotlib(
    'default', # the style theme
    {
        'font.size': 10,            # small font
        'figure.figsize': (10, 8),  # good dimensions
        'svg.fonttype': 'none'      # editable with notepad
    }
)


# Load FTMap files
for filename in glob("/home/peu/Desktop/Qualificação/TR_Dímero/test/*.pdb"):
    load_ftmap(
        filename,
        deep_search=False
    )

# Delete low strength hotspots, it's just noise!
for obj in pm.get_object_list('*.CS.* AND p.S<5'):
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
    # first object group is 1BZL, the site can be
    # given by a ligand or residue index of that 
    # structure also.
        #site='resi 200-210+187-195',  # from 200 to 210 and from 187 to 195
        #site='*',                     # all and every residue
        # site='resn GCG',             # probably absent in bare FTMove structures
        site='resi 436',               # only 5 A around residue 436
    # How many angstroms will be given around the
    # site? It can be 0 (exclusively resi 436),
    # 5 (a large padding), or any other value.
    site_radius=5,
    # How far from residues to look for probes.
    contact_radius=4.0,
    
    # Analysis Title.
    figure_title="Trypanothione Reductase (FTMove Analysis)",

    # Do plot the fingerprints.
    fingerprints_plot=True,
    # Fingerprints will have 40 labels.
    nbins=40,
    # Fingerprints will share the the x axis and
    # display residues from the first mentionated
    # conformation (1BZL).
    sharex=True,
    # Set the height of all fingerprints the same,
    # so they become easy to compare.
    share_ylim=True,

    # Do plot the HCA distance matrix (heatmap).
    heatmap_plot=True,
    # Annotate the heatmap with cell values.
    annotate=True,

    # Plot the HCA dendrogram.
    dendrogram_plot=True,
    # Ward is cool.
    linkage_method=LinkageMethod.WARD,
    # Do not cluster by height threshold.
    color_threshold=-1.0,
    # Neither specify the exact number of clusters.
    nclusters=-1,
)
plt.waitforbuttonpress()
