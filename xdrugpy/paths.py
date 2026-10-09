from pathlib import Path
from tempfile import mkdtemp

from pymol import Qt

QStandardPaths = Qt.QtCore.QStandardPaths
try:
    _data_location = QStandardPaths.AppLocalDataLocation
except AttributeError:
    try:
        _data_location = QStandardPaths.AppDataLocation
    except AttributeError:
        _data_location = QStandardPaths.StandardLocation.AppDataLocation
RESOURCES_DIR = Path(
    QStandardPaths.writableLocation(_data_location)
) / "XDrugPy"
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

LIGAND_LIBRARIES_DIR = Path(RESOURCES_DIR / "libs/ligands/")
LIGAND_LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)

RECEPTOR_LIBRARIES_DIR = Path(RESOURCES_DIR / "libs/receptors/")
RECEPTOR_LIBRARIES_DIR.mkdir(parents=True, exist_ok=True)

TEMPDIR = Path(mkdtemp(prefix="XDrugPy-"))


VERSION_FILE = Path(RESOURCES_DIR) / "version.txt"

