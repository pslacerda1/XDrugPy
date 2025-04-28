import os.path
from glob import glob
from pymol import cmd as pm
from xdrugpy.hotspots import (
    load_ftmap, ho, eftmap_overlap, _eftmap_overlap_get_aromatic, plot_hca,
    plot_heatmap, HeatmapFunction, fp_sim)
from xdrugpy.rmsf import rmsf
from xdrugpy.mapping import get_mapping
from xdrugpy.utils import expression_selector, multiple_expression_selector
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import json

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
    
    rmsf("*.K15_D_00", '*.protein', axis=img_gen)
    rmsf("*.K15_D_00", '*.protein', axis=img_ref)

    assert images_identical(img_ref, img_gen)


# def test_mapping():
#     pm.reinitialize()
#     load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
#     load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
#     load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')
#     load_ftmap(f"{pkg_data}/A7YT55_6css_atlas.pdb", 'eftmap')

#     mapping = get_mapping('eftmap', '*.protein')
#     g = mapping.groupby(['resi', 'chain'])


def test_eftmap_overlap():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/p38_1R39.pdb')
    deloc_xyz = _eftmap_overlap_get_aromatic('ligand')
    assert deloc_xyz.shape == (17, 3)

    deloc_contacts = eftmap_overlap('ligand', 'p38_1R39.ACS_aromatic_*')
    assert deloc_contacts == 12


def test_selector():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group')
    expr = '*K15_D_* S0<22'
    assert len(expression_selector(expr, type='K15')) == 4

    expr = "* S>=34"
    assert len(expression_selector(expr)) == 2
    
    expr = "MD<14 S0>=16"
    assert len(expression_selector(expr, 'K15')) == 1

    expr = "MD<14 S0>=16"
    assert len(expression_selector(expr, 'CS')) == 0

    expr = "*K15* CD<9 S0>=20"
    assert len(expression_selector(expr)) == 3

    expr = "*K15_[BD]* S0>=15 S0<=16"
    assert len(expression_selector(expr)) == 4

    expr = "group.K15_D_00"
    assert len(expression_selector(expr)) == 1

    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group_B')
    expr = "*.K15_D_00"
    assert len(expression_selector(expr)) == 2

    expr = "S==20"
    assert len(expression_selector(expr)) == 2

    expr = "S=20"
    assert len(expression_selector(expr)) == 2

    expr = "S>500"
    assert len(expression_selector(expr, type='CS')) == 0

    expr = "*.CS* S>13"
    assert len(expression_selector(expr, type='K15')) == 0
    

def test_multiple_selector():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group')
    expr = '*K15_D_* S0<22 : S==20' 
    result = multiple_expression_selector(expr)
    assert len(result) == 2
    assert result[0] == {'group.K15_D_00', 'group.K15_D_01', 'group.K15_D_02', 'group.K15_D_03'}
    assert result[1] == {'group.CS_00'}


def test_hca():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group')

    expr = "*.CS_* S>=5"
    img_ref = f'{pkg_data}/test_hca_ref.png'
    img_gen = f'{pkg_data}/test_hca_gen.png'

    dendro, medoids = plot_hca(expr, color_threshold=0.7, axis=img_gen)
    assert medoids['C1'].pop() == 'group.CS_02'
    assert medoids['C1'] == []
    assert images_identical(img_ref, img_gen)



def test_heatmap():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group')
    expr = "*.K15_* S0>=13"

    img_ref = f'{pkg_data}/test_heatmap_ref.png'
    img_gen = f'{pkg_data}/test_heatmap_gen.png'
    
    plot_heatmap(expr, method=HeatmapFunction.RESIDUE_JACCARD, axis=img_ref)
    plot_heatmap(expr, method=HeatmapFunction.RESIDUE_JACCARD, axis=img_gen)
    assert images_identical(img_ref, img_gen)


def test_fpt():
    pm.reinitialize()

    load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
    load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
    load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')

    img_gen1 = f'{pkg_data}/test_fpt1_gen.png'
    img_gen2 = f'{pkg_data}/test_fpt2_gen.png'

    img_ref1 = f'{pkg_data}/test_fpt1_ref.png'
    img_ref2 = f'{pkg_data}/test_fpt2_ref.png'

    fp_sim(
        "1dq8.K15_D_00 : 1dq9.K15_D_00 : 1dqa.K15_B_00",
        site="* within 4 of *_D_00",
        nbins=31,
        axis_fingerprint=img_gen1,
        axis_dendrogram=img_gen2,
    )

    assert images_identical(img_ref1, img_gen1)
    assert images_identical(img_ref2, img_gen2)
