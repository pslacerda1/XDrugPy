from glob import glob
from matplotlib import pyplot as plt
from pymol import cmd as pm

from xdrugpy.hotspots import (
    load_ftmap,
    fpt_sim,
    LinkageMethod,
)

# my opinion
plt.rcParams.update({
    'font.size': 14,
    # not very wide so fingerprints can be visualized
    'figure.figsize': (10, 8),

    # editable SVG text when opened on notepad
    'svg.fonttype': 'none'
})


# Load FTMap files
load_ftmap(glob(
    "/home/peu/Desktop/Qualificação/Kinases/*.pdb"
))

# Delete low strength hotspots, it's just noise!
for obj in pm.get_object_list('*.CS.* AND p.ST<5'):
    pm.delete(obj)

# Run the fingerprint similarity analisys
fpt_sim(
    # This long string has the wanted hotspots
    # separated by slashes. USE LESS STRUCTURES.
    multi_seles="""
        1JNK.CS.* / 1K3A.CS.* / 1PMN.CS.* /
        1E9H.CS.* / 1F5Q.CS.* / 1FIN.CS.* /
        1AGW.CS.* / 1FGI.CS.* / 1FGK.CS.*
    """,
    # Focus the analysis to a site. Because the
    # first object is 1JNK, the site can be given
    # by a ligand or residue index of that 
    # structure also.
        #site='resi 207-209+187-189',
        # site='*',
    site='pepseq DFG OR pepseq HRD',
    # How many angstroms will be given around the
    # site? It can be 0, 5, any value.
    site_radius=5,
    # Apply multiple sequence alignment
    seq_align_omega=True,
    # How far from residues to look for probes.
    contact_radius=4.0,
    # Do plot fingerprints.
    plot_fingerprints=True,
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
plt.show()