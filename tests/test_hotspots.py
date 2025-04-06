import os.path
from glob import glob
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap, ho, eftmap_overlap, _eftmap_overlap_get_aromatic, expression_selector


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

    expr = "S>500"
    assert len(expression_selector(expr, type='CS')) == 0
    