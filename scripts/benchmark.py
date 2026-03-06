from glob import glob
from xdrugpy.hotspots import load_ftmap

from pymol import cmd as pm

files = glob("tests/data/*.pdb")
for file in files[:5]:
    ftmap = load_ftmap(file)

assert pm.get_object_list("p.S0<20")
