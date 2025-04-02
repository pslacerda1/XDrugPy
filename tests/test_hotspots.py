import os.path
from glob import glob
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap, ho, eftmap_overlap, _eftmap_overlap_get_aromatic, plot_dendrogram


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

