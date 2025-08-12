from pymol import cmd as pm
from collections import defaultdict
from functools import lru_cache
import numpy as np
from fnmatch import fnmatch
from rcsbapi.search import SeqSimilarityQuery

from .utils import declare_command, mpl_axis



@declare_command
def fetch_similar(
    pdb_id: str,
    asm_id: int,
    identity_cutoff: float,
    site: str=None,
    site_margin: float = 4.0,
    max_entries:int = 50
):
    pdb_id = pdb_id.upper()
    obj0 = '%s-%s' % (pdb_id, asm_id)
    pm.fetch(pdb_id, obj0, type='pdb%s' % asm_id)
    seq = []
    pm.iterate(
        f'%{obj0} & guide & alt +A',
        'seq.append((chain, oneletter))',
        space=locals()
    )
    chain = None
    sequences = []
    for new_chain, oneletter in seq:
        if new_chain != chain:
            chain = new_chain
            sequences.append("")
        sequences[-1] += oneletter
    visited_chains = set()
    visited_objs = {obj0}
    for seq in sequences:
        if len(seq) <= 25 or seq in visited_chains:
            continue
        visited_chains.add(seq)
        query = SeqSimilarityQuery(
            seq,
            identity_cutoff=identity_cutoff,
            sequence_type="protein"
        )
        for asm in list(query("assembly"))[:max_entries]:
            asm = asm.split('-')
            pdb_id, asm_id = asm[0], asm[1]
            obj = '%s-%s' % (pdb_id, asm_id)
            if obj in visited_objs:
                continue
            visited_objs.add(obj)
            pm.fetch(pdb_id, obj, type='pdb%s' % asm_id)
            pm.cealign(obj0, obj)
            if not site:
                continue
            items = set()
            pm.iterate(
                f'(%{obj} & not (polymer | resn HOH)) within {site_margin} of ({site})',
                'items.add((resn, chain))',
                space=locals()
            )
            if items:
                pm.delete(obj)
                continue
            peptides = set()
            pm.iterate(
                f'(%{obj} & polymer) within {site_margin} of ({site})',
                'peptides.add(chain)',
                space=locals()
            )
            for chain in peptides:
                if pm.count_atoms(f'%{obj} & name CA & chain {chain}') < 25:
                    pm.delete(obj)
                    break
            else:
                pass


@declare_command
def rmsf(
    ref_site: str,
    prot_expr: str,
    site_margin:float = 3.0,
    qualifier: str = 'name CA',
    pretty: bool = True,
    axis: str = ''
):
    """
DESCRIPTION
    Calculate the RMSF of multiple related structures.

    A reference site must be supplied to focus, however full protein
    analysis can be achieved with a star * wildcard. A protein
    expression based on fnmatch to select the structures to calculate
    the RMSF must also be supplied.

OPTIONS
    ref_site: str
        A site expression to focus the RMSF calculation.
    prot_expr: str
        An expression to select the structures to calculate the RMSF.
    site_margin: float = 3.0
        The margin to consider the site around the ref_site.
    qualifier: str = 'name CA'
        A qualifier to select the atoms to calculate the RMSF.
    pretty: bool = True
        If True, it will show the RMSF in a pretty way. 


EXAMPLES
    # Calculate RMSF of site residues 10-150 for all proteins in the
    # session.
    rmsf resi 10-150, *.protein, pretty=False


    # Calculate RMSF of the full protein considering 1ABC and 2XYZ.
    rmsf *, 1ABC.protein 2XYZ.protein
    
    # Use all atoms instead of only alpha carbons.
    rmsf *, *.protein, qualifier=*

    """
    frames = []
    for obj in pm.get_object_list():
        if fnmatch(obj, prot_expr):
            frames.append(obj)

    f0 = frames[0]
    site = set()
    pm.iterate(
        f'(%{f0} & polymer) within {site_margin} of ({ref_site})',
        'site.add((resn,int(resi),chain))',
        space={'site': site}
    )
    
    @lru_cache(25000)
    def get_residues(frame):
        residues = []
        coords = np.empty((0, 3))
        chains = pm.get_chains(f"{frame} & polymer")
        sele = f"{frame} & {qualifier} & ("

        for i, chain in enumerate(chains):
            if i == 0:
                sele += f"(c. {chain} &"
            else:
                sele += f" | (c. {chain} &"
            idx_resids = "+".join(str(r[1]) for r in site if r[2] == chain)
            sele += f' i. {idx_resids}'
            sele += ") "
        sele += ")"
        
        for a in pm.get_model(sele).atom:
            resi = (a.resn, int(a.resi), a.chain)
            if resi in site:
                residues.append(resi)
                coords = np.vstack([coords, a.coord])
        return residues, coords

    # Aggregate coords from all frames
    X = {}
    for fr in frames:
        resis, coordinates = get_residues(fr)
        for resi, coords in zip(resis, coordinates):
            if resi not in X:
                X[resi] = np.empty((0,3))
            X[resi] = np.vstack([X[resi], coords])

    # Sort residues
    X = {k: X[k] for k in sorted(X, key=lambda z: (z[2], z[1]))}

    # Find mean positions for each reisude
    means = {}
    for resi in X:
        means[resi] = np.mean(X[resi], axis=0)

    # Calculate RMSF
    RMSF = []
    LABELS = []
    pm.alter(f"{f0} & polymer", "p.rmsf=0.0")
    for resi, coords in X.items():
        rmsf = np.sum((coords - means[resi]) ** 2) / coords.shape[0]
        rmsf = np.sqrt(rmsf)
        label = '%s %s:%s' % resi
        pm.alter(f"{f0} & i. {resi[1]} & c. {resi[2]}", f"p.rmsf={rmsf}")
        LABELS.append(label)
        RMSF.append(rmsf)
    
    # Show data
    if pretty:
        pm.hide('everything', f"{f0} & polymer")
        pm.show_as("line", f"{f0} & polymer")
        pm.spectrum("p.rmsf", "rainbow", f"{f0} & polymer")

    with mpl_axis(axis) as ax:
        ax.bar(LABELS, RMSF)
        ax.set_xlabel("Residue")
        ax.set_ylabel("RMSF")
        ax.tick_params(axis='x', rotation=90)
    
    return RMSF, LABELS


# @declare_command
# def rmsd_hca(
#     ref_site: str,
#     prot_expr: str,
#     qualifier: str = 'name CA',
#     site_margin: float = 5.0,
#     linkage_method: str = 'ward',
#     color_threshold: float = 0.0,
#     annotate: bool = True,
#     axis: str = ''
# ):
#     """
#     DESCRIPTION
#         Calculate the RMSD of multiple related structures. First it realizes
#         multiple sequence/structure alignment with the cealign function in
#         order to get the equivalent atoms, so it can be realized between
#         relatively distant homologues.

#         A reference site must be supplied to focus, however full protein
#         analysis can be achieved with a star * wildcard. A expression
#         based on fnmatch select the structures to calculate the RMSD.
#     """
#     frames = []
#     for obj in pm.get_object_list():
#         for expr in prot_expr.split():
#             if fnmatch(obj, expr):
#                 frames.append(obj)
#     f0 = frames[0]

#     site = set()
#     pm.iterate(
#         f'(%{f0} & polymer) within {site_margin} of ({ref_site})',
#         'site.add((resn,resi,chain))',
#         space={'site': site}
#     )
    
#     # Aggregate coords from all frames
#     X = []
#     for i1, f1 in enumerate(frames):
#         for i2, f2 in enumerate(frames):
#             if i1 >= i2:
#                 continue
#             rmsd = pm.rms(
#                 f"(%{f1} & polymer & {qualifier}) within {site_margin} of ({ref_site})",
#                 f"(%{f2} & polymer & {qualifier}) within {site_margin} of ({ref_site})",
#             )
#             X.append(rmsd)
#     X = np.array(X)
#     return plot_hca_base(max(X)-X, frames, linkage_method, color_threshold, annotate, axis)


from pymol import Qt

QWidget = Qt.QtWidgets.QWidget
QFileDialog = Qt.QtWidgets.QFileDialog
QFormLayout = Qt.QtWidgets.QFormLayout
QPushButton = Qt.QtWidgets.QPushButton
QSpinBox = Qt.QtWidgets.QSpinBox
QDoubleSpinBox = Qt.QtWidgets.QDoubleSpinBox
QLineEdit = Qt.QtWidgets.QLineEdit
QCheckBox = Qt.QtWidgets.QCheckBox
QVBoxLayout = Qt.QtWidgets.QVBoxLayout
QHBoxLayout = Qt.QtWidgets.QHBoxLayout
QDialog = Qt.QtWidgets.QDialog
QComboBox = Qt.QtWidgets.QComboBox
QTabWidget = Qt.QtWidgets.QTabWidget
QLabel = Qt.QtWidgets.QLabel
QTableWidget = Qt.QtWidgets.QTableWidget
QTableWidgetItem = Qt.QtWidgets.QTableWidgetItem
QGroupBox = Qt.QtWidgets.QGroupBox
QHeaderView = Qt.QtWidgets.QHeaderView

QtCore = Qt.QtCore
QIcon = Qt.QtGui.QIcon


class FinderWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QFormLayout()
        self.setLayout(layout)

        self.pdbLine = QLineEdit()
        layout.addRow("PDB:", self.pdbLine)

        self.assemblySpin = QSpinBox()
        self.assemblySpin.setValue(1)
        self.assemblySpin.setMinimum(1)
        self.assemblySpin.setMaximum(9)
        layout.addRow("Assembly:", self.assemblySpin)

        self.identityCutoffSpin = QDoubleSpinBox()
        self.identityCutoffSpin.setMinimum(0.300)
        self.identityCutoffSpin.setMaximum(1.000)
        self.identityCutoffSpin.setValue(0.900)
        self.identityCutoffSpin.setSingleStep(0.050)
        self.identityCutoffSpin.setDecimals(3)
        layout.addRow("Identity cutoff:", self.identityCutoffSpin)

        self.siteLine = QLineEdit()
        layout.addRow("Site:", self.siteLine)

        self.siteMarginSpin = QDoubleSpinBox()
        self.siteMarginSpin.setMinimum(1.0)
        self.siteMarginSpin.setMaximum(10.0)
        self.siteMarginSpin.setValue(3.0)
        self.siteMarginSpin.setSingleStep(0.5)
        self.siteMarginSpin.setDecimals(1)
        layout.addRow("Site margin:", self.siteMarginSpin)

        self.maxEntriesSpin = QSpinBox()
        self.maxEntriesSpin.setValue(50)
        self.maxEntriesSpin.setMinimum(2)
        self.maxEntriesSpin.setMaximum(9999)
        layout.addRow("Max entries:", self.maxEntriesSpin)

        fetchButton = QPushButton("Find")
        fetchButton.clicked.connect(self.find)
        layout.addWidget(fetchButton)

    def find(self):
        fetch_similar(
            pdb_id=self.pdbLine.text().strip(),
            asm_id=self.assemblySpin.value(),
            identity_cutoff=self.identityCutoffSpin.value(),
            site=self.siteLine.text().strip(),
            site_margin=self.siteMarginSpin.value(),
            max_entries=self.maxEntriesSpin.value(),
        )


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(300, 250)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("XDrugPy")

        tab = QTabWidget()
        tab.addTab(FinderWidget(), "Finder")

        layout.addWidget(tab)


dialog = None


def run_plugin_gui():
    global dialog
    if dialog is None:
        dialog = MainDialog()
    dialog.show()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("XDrugPy::Multi", run_plugin_gui)
