import subprocess
import pytest
from pymol import cmd as pm
from . import PKG_DATA_DIR


@pytest.fixture
def test_name(request):
    yield request.function.__name__


@pytest.fixture(scope='session', autouse=True)
def xdrugpy_install():
    try:
        subprocess.check_output(
            ['xdrugpy_xhf', '--help']
        )
        subprocess.check_output(
            ['vina', '--help']
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        import xdrugpy
        xdrugpy.xdrugpy_install()


@pytest.fixture(scope='session', autouse=True)
def configure_matplotlib():
    from xdrugpy import configure_matplotlib
    import numpy as np
    np.random.seed(42)
    configure_matplotlib(
        backend='svg',
        params={
            'font.sans-serif': ['DejaVu Sans Mono', 'Arial', 'Helvetica'],
            'font.family': 'sans-serif',
            'svg.fonttype': 'none',
            'svg.hashsalt': 'fixed_salt_123',
        }
    )

@pytest.fixture(scope='session')
def load_1dq9():
    from xdrugpy.hotspots import load_ftmap
    ftmap = load_ftmap(
        filename=PKG_DATA_DIR / "1dq9_atlas.pdb",
        group="deep_1dq9",
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap
    pm.delete('deep_1da9')


@pytest.fixture(scope='session')
def load_1dq8():
    from xdrugpy.hotspots import load_ftmap
    ftmap = load_ftmap(
        filename=PKG_DATA_DIR / "1dq8_atlas.pdb",
        group="deep_1dq8",
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap
    pm.delete('deep_1da8')


@pytest.fixture(scope='session')
def load_2tpr():
    from xdrugpy.hotspots import load_ftmap
    ftmap = load_ftmap(
        filename=PKG_DATA_DIR / "2TPR.pdb",
        group='deep_2tpr',
        deep_search=True,
        remove_nested=True,
    )
    yield ftmap
    pm.delete('deep_2tpr')


@pytest.fixture(scope='session')
def load_1byq_and_lbaf3():
    from xdrugpy.hotspots import load_ftmap
    load_ftmap(
        filename=PKG_DATA_DIR / "1BYQ.pdb",
        deep_search=False,
    )
    load_ftmap(
        filename=PKG_DATA_DIR / "LB_AF3.pdb",
        deep_search=False,
    )
    yield
    pm.delete('1BYQ LB_AF3')