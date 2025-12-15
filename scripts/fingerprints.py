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
    'figure.figsize': (10, 6),

    # editable SVG text when opened on notepad
    'svg.fonttype': 'none'
})


#----------     Scripting       ----------#
#
# This section was crafted by hand to allow
# customization.
#

# Load FTMap files
load_ftmap(glob(
    "/home/peu/Desktop/Qualificação/Kinases/*.pdb"
))

# Delete low strength hotspots, it's just noise!
for obj in pm.get_object_list('*.CS* AND p.ST<5'):
    pm.delete(obj)


# Run the fingerprint similarity analisys
_, _, dendro, _ = fpt_sim(
    # This long string has the wanted hotspots
    # separated by slashes
    multi_seles="""
        1JNK.CS* / 1K3A.CS* / 1PMN.CS* /
        1E9H.CS* / 1F5Q.CS* / 1FIN.CS* /
        1AGW.CS* / 1FGI.CS* / 1FGK.CS* /
        1A9U.CS* / 1BL6.CS* / 1BMK.CS* /
        1FVT.CS* / 1GIH.CS* / 1GII.CS* /
        1E1V.CS* / 1GIJ.CS* / 1GZ8.CS* /
        1G3N.CS* / 1IRK.CS* / 1T46.CS*
    """,
    # Focus the analysis to a site. Because the
    # first object is 1JNK, the site can be given
    # by a ligand or residue index also.
    #site='resi 207-209+187-189',
    site='pepseq DFG OR pepseq HRD',
    # How many angstroms will be given around the
    # site? It can be 0, 5, any value.
    site_radius=5,
    # Minimum accepted conservation to sequence
    # align.
    conservation='*:.',
    # How far from residues to look for probes?
    contact_radius=5.0,
    # Do plot fingerprints.
    plot_fingerprints=True,
    # Fingerprints will have 40 labels.
    nbins=40,
    # They will share the the x axis and display
    # residues from the first mentionated
    # conformation (1JNW).
    sharex=True,
    # Do plot the HCA.
    plot_hca=True,
    # Annotate the hetmap with cell values.
    annotate=True,
    # Do not cluster the HCA by height threshold.
    # Change it to 0.4 to enable the Scoring section.
    color_threshold=0.0,
    # Ward is cool.
    linkage_method=LinkageMethod.WARD,
)
# Close the plot windows after visualization to
# continue the analysis to the scoring section.
plt.show() 


#----------     Scoring       ----------#
#
# This section was mostly written by
# Gemini except by the DFG data filled in
# and general "it works" touch.
#
# It labels the unsupervised clusters with
# previous knew classes
#
from pprint import pp
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
DFG = {
    'DFGin_ABAminus':     ['1JNK', '1K3A', '1PMN'],
    'DFGin_BLAminus':     ['1E9H', '1F5Q', '1FIN'],
    'DFGin_BLAplus':      ['1AGW', '1FGI', '1FGK'],
    'DFGin_BLBminus':     ['1A9U', '1BL6', '1BMK'],
    'DFGin_BLBplus':      ['1FVT', '1GIH', '1GII'],
    'DFGint_BLBtrans':    ['1E1V', '1GIJ', '1GZ8'],
    # 'DFGinter_BABtrans':  ['1BYG', '1FVR', '2CLQ'],
    'DFGout_BBAminus':    ['1G3N', '1IRK', '1T46'],
}
y_true = []
y_pred = []
for lbl, clu in zip(dendro['ivl'], dendro['leaves_color_list']):
    lbl = lbl[:lbl.find('.')]
    for klass, candidates in DFG.items():
        for pdb_id in candidates:
            if pdb_id == lbl:
                y_true.append(klass)
                y_pred.append(clu)

ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
homogeneity = homogeneity_score(y_true, y_pred)
completeness = completeness_score(y_true, y_pred)
v_measure = v_measure_score(y_true, y_pred)

print(f"Adjusted Rand Index (ARI):     {ari:.4f}")
print(f"Normalized Mutual Info (NMI):  {nmi:.4f}")
print(f"Homogeneity Score:             {homogeneity:.4f}")
print(f"Completeness Score:            {completeness:.4f}")
print(f"V-Measure:                     {v_measure:.4f}")


# These were my results to unsupervised labeling
# structures of kinases, a complex protein family.

### Adjusted Rand Index (ARI):     0.5028
### Normalized Mutual Info (NMI):  0.7972
### Homogeneity Score:             0.8176
### Completeness Score:            0.7777
### V-Measure:                     0.7972
