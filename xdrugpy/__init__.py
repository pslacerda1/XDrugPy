from pymol import cmd as pm


@pm.extend
def xdrugpy_install(plugin_version):
    from . import install
    install.install_plugin(plugin_version)
    print("XDrugPy installation finished!")



from .public import *
from .public import __ALL__, __init_plugin__
