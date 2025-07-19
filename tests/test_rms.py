import os.path
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap
from xdrugpy.rms import rmsf, rmsd_hca
from PIL import Image
import numpy as np

pkg_data = os.path.dirname(__file__) + '/data'


def images_identical(img1_path, img2_path):
    """DeepSeek"""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    if img1.size != img2.size or img1.mode != img2.mode:
        return False
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return np.array_equal(arr1, arr2)


def test_rmsf():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
    load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
    load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')

    img_ref = f'{pkg_data}/test_rmsf_ref.png'
    img_gen = f'{pkg_data}/test_rmsf_gen.png'
    
    rmsf("*.K15_D_00", '*.protein', site_margin=5, axis=img_gen)
    assert images_identical(img_ref, img_gen)

def test_rmsd_hca():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
    load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
    load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')

    img_ref = f'{pkg_data}/test_rmsd_hca_ref.png'
    img_gen = f'{pkg_data}/test_rmsd_hca_gen.png'
    
    rmsd_hca("*", '*.protein', qualifier='*', site_margin=5, axis=img_gen)
    assert images_identical(img_ref, img_gen)