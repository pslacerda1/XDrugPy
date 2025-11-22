from os.path import expanduser
from glob import glob
from xdrugpy.hotspots import load_ftmap, plot_pairwise_hca
from pymol import cmd as pm
from matplotlib import pyplot as plt


# Lê as proteínas do FTMap, ajuste o caminho.
files = expanduser("~/Desktop/PEPTI/atlas/????_atlas.pdb")
for file in glob(files)[:25]:
    ftmap = load_ftmap(file)

    # A cada leitura, exclui hotspots que não antendem o critério de
    #  plotagem e tbm preserva estruturas (*.protein) da exclusão.
    for obj in pm.get_object_list("NOT (*.K15_* AND p.S0>16 OR *.protein)"):
        pm.delete(obj)


# Faz o HCA com hotspot overlap (HO) entre todos os hotspots
# que restaram após as leituras e exclusões.
plot_pairwise_hca(
    '*.K15_*',
    align=False,
    radius=4,
    color_threshold=.7,
    hide_threshold=True,
    annotate=False
)
plt.show()