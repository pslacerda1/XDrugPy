from os.path import dirname
from pprint import pp
from xdrugpy import (
    load_ftmap,
)

pkg_data = dirname(dirname(__file__)) + "/tests/data"
for pdb in ['1dqa', '1dq8', '1dq9']:
    filename = f"{pkg_data}/{pdb}_atlas.pdb"
    ftmap = load_ftmap(
        filename,
        group=pdb,
        deep_search=True,
        pretty=False
    )
    pp(ftmap)