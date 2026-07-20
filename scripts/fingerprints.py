from glob import glob
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

#
# Run the interaction (nearby atom count) fingerprint similarity analisys
#
calc_fingerprints(
    # This multi-line string contains, separated by
    # slashes, the desired hotspots whose atoms will
    # be counted.
    multi_seles="""
        1BZL.CS.* / 2JK6.CS.* / 2TPR.CS.* /
        2W0H.CS.* / 2WPF.CS.* / 6BU7.CS.*
    """,
    
    # Focus the analysis to a site. Because the
    # first object group is 1BZL, the site can be
    # given by a ligand or residue index of that 
    # structure also.
        #site='resi 200-210+187-195',  # from 200 to 210 and from 187 to 195
        #site='*',                     # all and every residue from 1BZL
        #site='my_object',             # for sure absent in FTMap structures
        site='resi 436',               # only 5 angstroms around residue 436
    # How many angstroms will be given around the
    # site? It can be 0 (exclusively resi 436),
    # 5 (a large padding), or any other value.
    site_radius=5.0,
    # How far from residues to look for probes.
    contact_radius=4.0,
    
    # Analysis title.
    figure_title=
        "Trypanothione Reductase\n(Interaction atom count fingerprint analysis)",

    # Do plot the fingerprints.
    fingerprints_plot=True,
    # Fingerprints will show up to 40 labels.
    nbins=40,
    # Fingerprints will share the the x axis and
    # display residues aligned from the first
    # mentionated conformation (1BZL).
    sharex=True,
    # Set the height of all fingerprints the same,
    # so they become easy to compare.
    share_ylim=True,

    #
    # Hierarchical Cluster Analysis (HCA)
    #

    # The distance of any two fingerprints is
    # given by the function d=1-s with the
    # similarity s=Pearson(fpt1,fpt2).Thats it,
    # two fingerprints are similiar if they are
    # correlated.
    
    # Chose the linkage method.
    linkage_method=LinkageMethod.AVERAGE,

    # Show only the labels of cluster medoids
    # (similar to centroids). Hide most labels
    # so the medoids are nitide in case of many
    # labels. Only works if clustering is enabled.
    only_medoids=False,

    # Do plot the HCA dendrogram tree.
    # There are two mutually exclusive options to cluster the
    # tree. In both options you may want to do a first run to
    # inspect the plot visually and then adjust the parameters.
    # They are:
    #    (i) by the height threshold which the colors will show up.
    #   (ii) by the number of desired clusters if it is already knew.
    dendrogram_plot=True,
        #color_threshold=-1.0,   # the threshold is disabled if set to -1.0
        nclusters=-1,            # the option is disabled if set to -1 or 0
    
    # Do plot the HCA distance matrix heatmap. Change
    # the filename by True or change the ending from
    # .svg to .png, for instance. Set to False to not plot.
    heatmap_plot='./tests/data/test_my_fpt_heatmap_gen.svg',
    # Annotate the heatmap with the values (not
    # only colors).
    annotate=True,
)
plt.waitforbuttonpress()
