import os.path
import matplotlib as mpl
import numpy as np
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap
from xdrugpy.multi import rmsf, fetch_similar


mpl.rcParams["svg.hashsalt"] = "42"
mpl.rcParams["svg.id"] = "42"
np.random.seed(42)

pkg_data = os.path.dirname(__file__) + "/data"



def images_identical(img1_path, img2_path):
    from PIL import ImageChops, Image
    import io
    import cairosvg

    def rasterize(svg_path):
        png_data = cairosvg.svg2png(url=svg_path)
        return Image.open(io.BytesIO(png_data)).convert("RGB")

    def compare_visual(svg1, svg2):
        img1 = rasterize(svg1)
        img2 = rasterize(svg2)
        diff = ImageChops.difference(img1, img2)
        return diff.getbbox() is None  # True if identical

    return compare_visual(svg1=img1_path, svg2=img2_path)


def test_rmsf():
    pm.reinitialize()
    load_ftmap(f"{pkg_data}/1dq8_atlas.pdb", "1dq8")
    load_ftmap(f"{pkg_data}/1dq9_atlas.pdb", "1dq9")
    load_ftmap(f"{pkg_data}/1dqa_atlas.pdb", "1dqa")
    img_ref = f"{pkg_data}/test_rmsf_ref.svg"
    img_gen = f"{pkg_data}/test_rmsf_gen.svg"
    rmsf("*.protein", "*.K15_D_00", site_margin=5, axis=img_gen)
    assert images_identical(img_ref, img_gen)

LIST_ITEM1 = {'pdb_id': '1E7W', 'asm_id': '1', 'resn': 'MTX', 'resi': '301', 'chain_id': 'A'}
LIST_ITEM2 = {'pdb_id': '1B0H', 'asm_id': '1', 'resn': None, 'resi': None, 'chain_id': 'B'}

def test_fetch_similar_0():
    pm.reinitialize()
    pm.fetch('1E92')
    data = fetch_similar(f'1E92', "protein", 0.9, 'resn HBI and chain A', max_results=3)
    assert len(pm.get_object_list()) == 1
    assert ('1W0C', 1, 'A', 'TAQ', 301) in data
    assert ('1E7W', 1, 'B', 'MTX', 301) in data

def test_fetch_similar_1():
    pm.reinitialize()
    pm.fetch("1e92")
    data = fetch_similar("1E92", "protein", 0.9, max_results=20)
    assert len(pm.get_object_list()) >= 17
    assert len(data) >= 16
