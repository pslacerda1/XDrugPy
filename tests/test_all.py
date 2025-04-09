import os.path
from glob import glob
from pymol import cmd as pm
from xdrugpy.hotspots import (
    load_ftmap, ho, eftmap_overlap, _eftmap_overlap_get_aromatic,
    expression_selector, multiple_expression_selector, plot_dendrogram,
    plot_heatmap, HeatmapFunction)
from xdrugpy.rmsf import rmsf
from matplotlib import pyplot as plt
import PIL.Image, PIL.ImageChops


pkg_data = os.path.dirname(__file__) + '/data'

def test_ho():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group')
    assert ho('*D_01 *D_00', verbose=False) == 371
    assert ho('*D_00 *D_01', verbose=False) == 371
    assert ho('*D_01 *D_00 *B_05', verbose=False) == 0
    count = ho('*D_01 *D_00', output_sele='__a', verbose=False)
    assert pm.count_atoms('__a') == count
    count = ho('*D_00 *B_04', output_sele='__a', verbose=False)
    assert pm.count_atoms('*D_00') > pm.count_atoms('*B_04') > count


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
    expr = '*K15_D_* S0<22 ; S==20' 
    result = multiple_expression_selector(expr)
    assert len(result) == 2
    assert result[0] == {'group.K15_D_00', 'group.K15_D_01', 'group.K15_D_02', 'group.K15_D_03'}
    assert result[1] == {'group.CS_00'}


def test_dendrograma():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group')

    expr = "*.CS_* S>=13"
    img_ref = f'{pkg_data}/test_dendrograma_ref.png'
    img_gen = f'{pkg_data}/test_dendrograma_gen.png'
    try:
        os.unlink(img_gen)
    except:
        pass

    plot_dendrogram(expr, residue_align=False, ax=img_gen)
    
    ref = PIL.Image.open(img_ref)
    gen = PIL.Image.open(img_gen)
    diff = PIL.ImageChops.difference(ref, gen)
    assert not diff.getbbox()

def test_heatmap():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/A7YT55_6css_atlas.pdb', 'group')
    expr = "*.K15_* S0>=13"

    img_ref = f'{pkg_data}/test_heatmap_ref.png'
    img_gen = f'{pkg_data}/test_heatmap_gen.png'
    try:
        os.unlink(img_gen)
    except:
        pass
    
    plot_heatmap(expr, method=HeatmapFunction.RESIDUE_JACCARD, ax=img_gen)

    ref = PIL.Image.open(img_ref)
    gen = PIL.Image.open(img_gen)
    diff = PIL.ImageChops.difference(ref, gen)
    assert not diff.getbbox()




def test_rmsf():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
    load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
    load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')

    img_ref = f'{pkg_data}/test_rmsf_ref'
    img_gen = f'{pkg_data}/test_rmsf_gen'

    rmsf("*.K15_D_00", '*.protein', ax=img_gen)
    rmsf("*.K15_D_00", '*.protein', ax=img_ref)

    ref = PIL.Image.open(img_ref)
    gen = PIL.Image.open(img_gen)
    diff = PIL.ImageChops.difference(ref, gen)
    assert not diff.getbbox()
