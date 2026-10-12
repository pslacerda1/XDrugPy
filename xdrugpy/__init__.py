from pymol import Qt
from pymol import cmd as pm

from .install import UpdaterDialog, install_plugin


#
# Expose a command line interface to version updater
#
@pm.extend
def xdrugpy_install(plugin_version):
    install_plugin(plugin_version)
    print("XDrugPy installation finished!")


#
# Expose almost everything from public
#

try:
    from .public import *
    from .public import __ALL__
except ImportError:
    import traceback
    traceback.print_exc()
    print("XDrugPy installation failed!")


_updater_dialog = None

def __init_plugin__(app=None):
    global _updater_dialog
    if _updater_dialog is None:

        QLocale = Qt.QtCore.QLocale
        locale = QLocale(QLocale.English, QLocale.Zimbabwe)
        QLocale.setDefault(locale)

        _updater_dialog = UpdaterDialog()

        from . import public
        public.__init_plugin__()
        
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("(XDrugPy) Settings", _updater_dialog.show)
