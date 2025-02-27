from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap, ho


def test_ho():
    load_ftmap('data/A7YT55_6css_atlas.pdb', 'group')
    assert ho('*D_01 *D_00', verbose=False) == 371
    assert ho('*D_00 *D_01', verbose=False) == 371
    assert ho('*D_01 *D_00 *B_05', verbose=False) == 0
    count = ho('*D_01 *D_00', output_sele='__a', verbose=False)
    assert pm.count_atoms('__a') == count
    count = ho('*D_00 *B_04', output_sele='__a', verbose=False)
    assert pm.count_atoms('*D_00') > pm.count_atoms('*B_04') > count
