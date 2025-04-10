import os.path
from pymol import cmd as pm
from xdrugpy.hotspots import load_ftmap
from xdrugpy.rmsf import rmsf
import PIL.Image, PIL.ImageChops


pkg_data = os.path.dirname(__file__) + '/data'

def test_rmsf():
    pm.reinitialize()
    load_ftmap(f'{pkg_data}/1dq8_atlas.pdb', '1dq8')
    load_ftmap(f'{pkg_data}/1dq9_atlas.pdb', '1dq9')
    load_ftmap(f'{pkg_data}/1dqa_atlas.pdb', '1dqa')

    img_ref = f'{pkg_data}/test_rmsf_ref.png'
    img_gen = f'{pkg_data}/test_rmsf_gen.png'
    
    rmsf("*.K15_D_00", '*.protein', axis=img_gen)
    rmsf("*.K15_D_00", '*.protein', axis=img_ref)

    ref = PIL.Image.open(img_ref)
    gen = PIL.Image.open(img_gen)
    diff = PIL.ImageChops.difference(ref, gen)
    assert not diff.getbbox()
