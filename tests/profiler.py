from pprint import pp
from os.path import expanduser
from glob import glob
from xdrugpy.hotspots import load_ftmap, plot_pairwise_hca

files = expanduser("~/Desktop/PEPTI/atlas/????_atlas.pdb")
files = glob(files)[:20]
ftmap = load_ftmap(files)

result = plot_pairwise_hca(
    '*.K15_*_00 AND p.ST > 10',
    align=False,
    radius=4,
    color_threshold=.7,
    hide_threshold=True,
    annotate=False,
    axis="tests/data/profiler.png"
)

