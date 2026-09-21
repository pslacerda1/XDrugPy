import pytest
from pymol import cmd as pm

from xdrugpy.hotspots import load_ftmap

from . import PKG_DATA_DIR


@pytest.fixture
def test_name(request):
    yield request.function.__name__


@pytest.fixture(scope='session')
def load_default_1dq8():
    ftmap_1dq8 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq8_atlas.pdb",
        group="default_1dq8",
        deep_search=False
    )
    yield ftmap_1dq8
    pm.delete('default_1da8')


@pytest.fixture(scope='session')
def load_default_1dq9():
    ftmap_1dq8 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq9_atlas.pdb",
        group="default_1dq9",
        deep_search=False
    )
    yield ftmap_1dq8
    pm.delete('default_1da9')


@pytest.fixture(scope='session')
def load_deep_1dq9():
    ftmap_1dq9 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq9_atlas.pdb",
        group="deep_1dq9",
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap_1dq9
    pm.delete('deep_1da9')


@pytest.fixture(scope='session')
def load_deep_1dq8():
    ftmap_1dq8 = load_ftmap(
        filename=PKG_DATA_DIR / "1dq8_atlas.pdb",
        group="deep_1dq8",
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap_1dq8
    pm.delete('deep_1da8')


@pytest.fixture(scope='session')
def load_deep_2tpr():
    ftmap = load_ftmap(
        filename=f"{PKG_DATA_DIR}/2TPR.pdb",
        group='deep_2tpr',
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap
    pm.delete('deep_2tpr')
