#
# Sample script to load DHODH structures and their corresponding FTMap results.
#

from os import chdir
from xdrugpy.hotspots import load_ftmap
from pymol import cmd as pm


chdir('~/Desktop/DHODH/')
data = [
    (
        '229', '229.pdb',
        'LbDHODH_TAP229_refine_60.pdb',
        'TXX', 131, 'B'
    ),
    (
        '249', '249.pdb',
        'LbDHODH_TAP249_refine_142.pdb',
        'CLC', 131, 'B'
    ),
    (
        '103', '103.pdb',
        'LbDHODH_pip103_DIMER.pdb',
        'CXY', 131, 'A'
    ),
]


for title, ftmap_pdb, crystal_pdb, resn, resid, chain in data:
    load_ftmap(
        ftmap_pdb,
        title,
        cd_to_anchor=False,
        combinatory_search=True,
        allow_nested=True
    )
    pm.load(crystal_pdb, 'temp')
    pm.align(
        f'%temp & c. {chain}',
        f'{title}.protein'
    )
    pm.extract(
        f'{title}.{resn}',
        f'%temp & resn {resn} & i. {resid} & c. {chain}'
    )
    pm.delete('temp')
