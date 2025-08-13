import os.path
import matplotlib as mpl
import numpy as np
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap
from xdrugpy.multi import rmsf, fetch_similar


mpl.rcParams['svg.hashsalt'] = '42'
mpl.rcParams['svg.id'] = '42'
np.random.seed(42)

pkg_data = os.path.dirname(__file__) + '/data'



def images_identical(img1_path, img2_path):
    with open(img1_path) as f1, open(img2_path) as f2:
        f1 = [l for l in f1.readlines() if '<dc:date>' not in l]
        f2 = [l for l in f2.readlines() if '<dc:date>' not in l]
        for l1, l2 in zip(f1, f2):
            if l1 != l2:
                return False
    return True


def test_rmsf():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
    load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
    load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')
    img_ref = f'{pkg_data}/test_rmsf_ref.svg'
    img_gen = f'{pkg_data}/test_rmsf_gen.svg'
    rmsf('*.protein', "*.K15_D_00", site_margin=5, axis=img_gen)
    assert images_identical(img_ref, img_gen)

def test_rmsf_slow():
    pm.reinitialize()
    img_ref = f'{pkg_data}/test_rmsf_slow_ref.svg'
    img_gen = f'{pkg_data}/test_rmsf_slow_gen.svg'
    fetch_similar('1e92', 1, 0.9, max_entries=20)
    assert len(pm.get_object_list()) == 16
    rmsf('*', "resn HBI", site_margin=5, axis=img_gen)
    assert images_identical(img_ref, img_gen)


def test_fetch_similar():
    pm.reinitialize()
    fetch_similar('1e92', 1, 0.9, max_entries=20)
    assert len(pm.get_object_list()) == 16

    pm.reinitialize()
    fetch_similar('1e92', 1, 0.9, site='resn HBI and chain A', max_entries=20)
    assert len(pm.get_object_list()) == 1

    pm.reinitialize()
    fetch_similar('1b5h', 1, 0.9, site='chain B', max_entries=50)
    assert len(pm.get_object_list()) == 3

# def test_rmsd_hca():
#     pm.reinitialize()
#     load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
#     load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
#     load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')

#     img_ref = f'{pkg_data}/test_rmsd_hca_ref.svg'
#     img_gen = f'{pkg_data}/test_rmsd_hca_gen.svg'
    
#     rmsd_hca("*", '*.protein', qualifier='name CA', site_margin=4, annotate=True, axis=img_gen)
#     assert images_identical(img_ref, img_gen)