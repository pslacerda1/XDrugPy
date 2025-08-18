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
    with open(img1_path) as f1, open(img2_path) as f2:
        f1 = [l for l in f1.readlines() if "<dc:date>" not in l]
        f2 = [l for l in f2.readlines() if "<dc:date>" not in l]
        for l1, l2 in zip(f1, f2):
            if l1 != l2:
                return False
    return True


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
    data = fetch_similar('1e92', 1, 0.9, 'resn HBI and chain A', max_entries=3)
    assert len(pm.get_object_list()) == 1
    assert  '1E7W' in data and all(data['1E7W']['1']['A'][k] == LIST_ITEM1[k] for k in ['pdb_id', 'asm_id', 'resn', 'resi', 'chain_id'])

def test_fetch_similar_1():
    pm.reinitialize()
    data = fetch_similar("1e92", 1, 0.9, max_entries=20)
    assert len(pm.get_object_list()) == 16
    assert not data

def test_fetch_similar_2():
    pm.reinitialize()
    data = fetch_similar("1e92", 1, 0.9, unbound_site="resn HBI and chain A", max_entries=3)
    assert len(pm.get_object_list()) == 1
    assert  '1E7W' in data and all(data['1E7W']['1']['A'][k] == LIST_ITEM1[k] for k in ['pdb_id', 'asm_id', 'resn', 'resi', 'chain_id'])

def test_fetch_similar_3():
    pm.reinitialize()
    data = fetch_similar("1bzl", 1, 0.9, max_entries=50)
    assert len(pm.get_object_list()) == 6
    assert not data

def test_fetch_similar_4():
    pm.reinitialize()
    data = fetch_similar("1b5h", 1, 0.9, unbound_site="chain B", max_entries=3)
    assert  '1B0H' in data and all(data['1B0H']['1']['B'][k] == LIST_ITEM2[k] for k in ['pdb_id', 'asm_id', 'resn', 'resi', 'chain_id'])
