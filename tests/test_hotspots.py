from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap, ho


def test_ho():
    ftmap = load_ftmap('data/A7YT55_6css_atlas.pdb', 'group')
    assert ho('*D_01 *D_00', verbose=False) == 1
    assert ho('*D_01 *D_00 *B_05', verbose=False) == 0
    assert ho('*D_01 *D_00', output_sele='__a', verbose=False)
    assert pm.count_atoms('__a') == 184
    assert ho('*D_00 *D_01', output_sele='__a', verbose=False)
    assert ho('*D_00 *D_01', output_sele='__a', verbose=False)
    assert pm.count_atoms('__a') == 184
    assert pm.count_atoms('*D_00') > 184