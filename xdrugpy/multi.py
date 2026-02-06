import time
import logging
from pymol import cmd as pm
import pandas as pd
import numpy as np
from fnmatch import fnmatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import OPTICS
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
from strenum import StrEnum
from pymol import CmdException

from .utils import (
    new_command,
    mpl_axis,
    PyMOLComboObjectBox,
    AligMethod,
    clustal_omega
)


logging.getLogger("rcsbapi").setLevel(logging.CRITICAL)
logging.getLogger("rcsbapi.search").setLevel(logging.CRITICAL)
logging.getLogger("rcsbapi.graphql").setLevel(logging.CRITICAL)


PROSTHETIC_GROUPS = "HEM FAD NAP NDP ADP FMN"
PROSTHETIC_GROUPS += " EDO PGE PEG GOL ACT"
PROSTHETIC_GROUPS += " HOH SO4"

RMSF_DEFAULT_QUALIFIER = "name CA"



class SequenceType(StrEnum):
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"


@new_command
def fetch_similar(
    sequence_sele: str,
    sequence_type: SequenceType = SequenceType.PROTEIN,
    identity_cutoff: float = 0.8,
    check_ligands: bool = False,
    check_peptides: bool = False,
    site_sele: str = None,
    site_margin: float = 4.0,
    max_peptide_length: int = 25,
    align_method: AligMethod = AligMethod.CEALIGN,
    ignore_ligands: str = PROSTHETIC_GROUPS,
    max_results: int = 50,
    fetch_extra: bool = False,
):
    seq = []
    pm.iterate(
        f"{sequence_sele} & polymer & name CA",
        "seq.append((chain, oneletter))",
        space=locals()
    )
    chain = None
    sequences = []
    for new_chain, oneletter in seq:
        if new_chain != chain:
            chain = new_chain
            sequences.append("")
        sequences[-1] += oneletter
    
    results = {}
    for seq in set(sequences):
        if len(seq) < 25:
            continue
        try:
            from rcsbapi.search import SeqSimilarityQuery
            query = SeqSimilarityQuery(
                seq,
                identity_cutoff=identity_cutoff,
                sequence_type=str(sequence_type)
            )
            assembly_ids = list(query(return_type="assembly"))[:max_results]
            time.sleep(0.1)  # rate limit
        except Exception as exc:
            print(f"Error querying sequence: {exc}")
            raise exc

        object_list = [
            o.upper()
            for o in pm.get_object_list(sequence_sele)
        ]
        cnt_asm_ids = Counter([
            asm.split('-')[0]
            for asm in assembly_ids
        ])
        for asm in assembly_ids:
            pdb_id, asm_id = asm.split('-')
            asm_id = int(asm_id)
            if cnt_asm_ids[pdb_id] == 1:
                obj = pdb_id
            else:
                obj = '%s_%s' % (pdb_id, asm_id)
            
            if obj.upper() in object_list:
                continue
            
            try:
                pm.fetch(pdb_id, obj, type="pdb%s" % asm_id)
            except (Exception, CmdException) as exc:
                print(f'Failed to download PDB entry {pdb_id} with Assembly {asm_id}: {exc}')
                continue

            ligands = set()
            peptides = set()
            organisms = set()
            results[(pdb_id, int(asm_id))] = {
                'organisms': organisms,
                'ligands': ligands,
                'peptides': peptides
            }

            # Start structural analysis
            try:
                pm.extra_fit(
                    selection=obj,
                    reference=sequence_sele,
                    method=str(align_method),
                    quiet=False
                )
            except Exception as exc:
                print(f'Failed to align an object entry: {obj}')
                continue

            if check_ligands:
                # Ligands were required to be explicitly checked
                mols = '+'.join(ignore_ligands.split())
                sele = f"(%{obj} AND NOT (polymer OR resn {mols})) NEAR_TO {site_margin} OF ({site_sele})"
                for at in pm.get_model(sele).atom:
                    ligands.add((at.resn, at.chain, int(at.resi)))
            
            if check_peptides:
                # Peptides of length <=25 are deemed ligands
                this_peptides = set()
                pm.iterate(
                    f"(%{obj} & polymer) NEAR_TO {site_margin} OF ({site_sele})",
                    "this_peptides.add(chain)",
                    space=locals(),
                )
                for chain in this_peptides:
                    if length := pm.count_atoms(f"name CA & (bymolecule (%{obj} & chain {chain}))") < max_peptide_length:
                        peptides.add((chain, length))
    # pm.order(assembly_ids, True)  # XXX segfaults
    # Fetch extra data
    if fetch_extra:
        from rcsbapi.data import DataQuery
        from collections import defaultdict
        pdb_id_list = list(set(pdb for pdb, _ in results))
        query = DataQuery(
            input_type="entries",
            input_ids=pdb_id_list,
            return_data_list=[
                "rcsb_entry_container_identifiers.entry_id",
                "polymer_entities.rcsb_polymer_entity.pdbx_description",
                "polymer_entities.rcsb_entity_source_organism.ncbi_scientific_name",
            ]
        )
        organisms = defaultdict(set)
        for pdb_id, asm_id in results.keys():
            res = query.exec()
            for entry in res['data']['entries']:
                if entry['rcsb_id'].upper() == pdb_id.upper():
                    for entity in entry['polymer_entities']:
                        desc = entity['rcsb_polymer_entity']['pdbx_description']
                        for org in entity['rcsb_entity_source_organism']:
                            org = org['ncbi_scientific_name']
                            organisms[(pdb_id, asm_id)].add((desc, org))
        for (pdb_id, asm_id), retval in results.items():
            retval['organisms'] = organisms[(pdb_id, asm_id)]
    return results


@new_command
def rmsf(
    prot_expr: str,
    ref_site: str = "*",
    site_radius: float = 4.0,
    align_method: AligMethod = AligMethod.CEALIGN,
    qualifier: str = RMSF_DEFAULT_QUALIFIER,
    omega_conservation: str = "*:.",
    pretty: bool = True,
    axis: str = "",
    quiet: bool = False,
):
    """
    DESCRIPTION
        Calculate the RMSF of multiple related structures.

        A reference site must be supplied to focus, however full protein
        analysis can be achieved with a star * wildcard. A protein
        expression based on fnmatch to select the structures to calculate
        the RMSF must also be supplied.

    OPTIONS
        prot_expr:
            An expression to select the structures to calculate the RMSF.
        ref_site:
            A site expression to focus the RMSF calculation.
        site_radius:
            The margin to consider the site around the ref_site.
        qualifier:
            A qualifier to select the atoms to calculate the RMSF.
        pretty:
            If True, it will show the RMSF in a pretty way.
    """
    frames = []
    for obj in pm.get_object_list(prot_expr):
            frames.append(obj)
    
    f0 = frames[0]
    site_sele = f"{f0} & polymer & ({f0} within {site_radius} of ({ref_site}))"
    if not quiet:
        print(f'Reference selection: "{site_sele}"')
    site_resis = []
    for at in pm.get_model(f"({site_sele}) & present & guide & polymer").atom:
        site_resis.append((at.model, at.index))
    if not quiet:
        print(f"Aligning structures to {f0} with method {align_method}...")
    try:
        pm.extra_fit(
            selection=' '.join(frames[1:]),
            reference=site_sele,
            method=str(align_method),
            quiet=False
        )
    except (Exception, CmdException) as exc:
        raise Exception(f"Failed to align objects to {f0}.") from exc

    mapping = clustal_omega(frames, omega_conservation)
    f0_map = mapping[f0]
    X = {}
    for frame, map in zip(frames, mapping.values()):
        for ref_res, res in zip(f0_map, map):
            if (f0, ref_res.index) not in site_resis:
                continue
            lbl = (ref_res.resn, ref_res.conservation, ref_res.resi, ref_res.chain)
            coords = pm.get_coords(
                f"({qualifier}) & (byres %{frame} & index {res.index})"
            )
            if lbl not in X:
                X[lbl] = []
            if coords is None or len(coords) == 0:
                print(f"Warning: no coordinates found for {frame} residue {res.resn} {res.resi}_{res.chain}")
                continue
            X[lbl].extend(coords)

    # Sort residues
    X = {k: np.array(X[k]) for k in sorted(X, key=lambda z: (z[3], z[2]))}

    # Calculate RMSF
    RMSF = []
    LABELS = []
    pm.alter(f"{f0} & polymer", "p.rmsf=0.0")
    for resi, coords in X.items():
        diff = coords - np.mean(coords, axis=0)
        squared_dist = np.sum(diff**2, axis=1)
        rmsf = np.sqrt(np.mean(squared_dist, axis=0))
        label = "%s%s %s:%s" % resi
        pm.alter(f"{f0} & i. {resi[2]} & c. {resi[3]}", f"p.rmsf={rmsf}")
        LABELS.append(label)
        RMSF.append(rmsf)

    # Show data
    if pretty:
        pm.hide("everything", site_sele)
        pm.show_as("line", site_sele)
        pm.spectrum("p.rmsf", "rainbow", site_sele)

    with mpl_axis(axis, constrained_layout=True) as ax:
        ax.bar(LABELS, RMSF)
        ax.set_ylabel("RMSF")
        ax.tick_params(axis="x", rotation=90)

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
QTreeWidget = Qt.QtWidgets.QTreeWidget
QTreeWidgetItem = Qt.QtWidgets.QTreeWidgetItem

QtCore = Qt.QtCore
QIcon = Qt.QtGui.QIcon


class FechSimilarWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = self.layout = QFormLayout()
        self.setLayout(layout)
        
        self.sequenceCombo = PyMOLComboObjectBox()
        layout.addRow("Query sequence:", self.sequenceCombo)

        self.sequenceTypeCombo = QComboBox()
        self.sequenceTypeCombo.addItems(list(map(str, SequenceType)))
        self.sequenceTypeCombo.setCurrentText(SequenceType.PROTEIN)
        layout.addRow("Sequence type:", self.sequenceTypeCombo)

        self.identityCutoffSpin = QDoubleSpinBox()
        self.identityCutoffSpin.setMinimum(0.300)
        self.identityCutoffSpin.setMaximum(1.000)
        self.identityCutoffSpin.setValue(0.900)
        self.identityCutoffSpin.setSingleStep(0.050)
        self.identityCutoffSpin.setDecimals(3)
        layout.addRow("Identity cutoff:", self.identityCutoffSpin)

        self.checkLigandsCheck = QCheckBox()
        self.checkLigandsCheck.setChecked(False)
        layout.addRow("Check ligands", self.checkLigandsCheck)

        self.checkPeptidesCheck = QCheckBox()
        self.checkPeptidesCheck.setChecked(False)
        layout.addRow("Check peptides", self.checkPeptidesCheck)

        self.siteLine = QLineEdit("")
        layout.addRow("Ligands site:", self.siteLine)

        self.siteMarginSpin = QDoubleSpinBox()
        self.siteMarginSpin.setMinimum(1.0)
        self.siteMarginSpin.setMaximum(10.0)
        self.siteMarginSpin.setValue(4.0)
        self.siteMarginSpin.setSingleStep(0.5)
        self.siteMarginSpin.setDecimals(1)
        layout.addRow("Site margin:", self.siteMarginSpin)

        self.ignoreLigandsLine = QLineEdit(PROSTHETIC_GROUPS)
        layout.addRow("Ignore ligands:", self.ignoreLigandsLine)
        
        self.alignMethodCombo = QComboBox()
        self.alignMethodCombo.addItems(list(map(str, AligMethod)))
        self.alignMethodCombo.setCurrentText(AligMethod.CEALIGN)
        layout.addRow("Align method:", self.alignMethodCombo)

        self.maxResultsSpin = QSpinBox()
        self.maxResultsSpin.setValue(50)
        self.maxResultsSpin.setMinimum(2)
        self.maxResultsSpin.setMaximum(9999)
        layout.addRow("Max entries:", self.maxResultsSpin)

        self.extraCheck = QCheckBox()
        layout.addRow("Fetch extra: ", self.extraCheck)

        findButton = QPushButton("Find")
        findButton.clicked.connect(self.find)
        layout.addWidget(findButton)
    
    def find(self):
        data = fetch_similar(
            sequence_sele=self.sequenceCombo.currentText(),
            sequence_type=SequenceType(self.sequenceTypeCombo.currentText()),
            identity_cutoff=self.identityCutoffSpin.value(),
            check_ligands=self.checkLigandsCheck.isChecked(),
            check_peptides=self.checkPeptidesCheck.isChecked(),
            site_sele=self.siteLine.text().strip(),
            site_margin=self.siteMarginSpin.value(),
            ignore_ligands=self.ignoreLigandsLine.text().strip(),
            align_method=AligMethod(self.alignMethodCombo.currentText()),
            max_results=self.maxResultsSpin.value(),
            fetch_extra=self.extraCheck.isChecked(),
        )
        self.dialog = FechSimilarResultsDialog(data)
        self.dialog.show()


class FechSimilarResultsDialog(QDialog):

    def __init__(self, data):
        super().__init__()
        self.setWindowTitle("Fetch Similar Results")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.header().hide()
        self.tree.header().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.tree)
        
        for pdb_id, asm_id in data.keys():
            pdb_item = QTreeWidgetItem([f"PDB {pdb_id} Assembly {asm_id}", "", ""])
            self.tree.addTopLevelItem(pdb_item)

            for org, macromol in data[(pdb_id, asm_id)]['organisms']:
                orgItem = QTreeWidgetItem([org, macromol, ""])
                pdb_item.addChild(orgItem)
            
            ligands = list(data[(pdb_id, asm_id)]['ligands'])
            visited = set()
            for resn, _, _, in ligands:
                    if resn in visited:
                        continue
                    visited.add(resn)
                    lig_item = QTreeWidgetItem([f"Residue {resn}", '', ''])
                    pdb_item.addChild(lig_item)
            if not visited:
                lig_item = QTreeWidgetItem(["No ligand found at site.", '', ''])
                pdb_item.addChild(lig_item)
        self.tree.expandAll()

    
class RmsfWidget(QWidget):

    def __init__(self):
        super().__init__()

        layout = QFormLayout()
        self.setLayout(layout)

        self.proteinExprLine = QLineEdit()
        layout.addRow("Proteins expression:", self.proteinExprLine)

        self.refSiteLine = QLineEdit("*")
        layout.addRow("Reference site:", self.refSiteLine)

        self.siteRadiusSpin = QDoubleSpinBox()
        self.siteRadiusSpin.setMinimum(1.0)
        self.siteRadiusSpin.setMaximum(10.0)
        self.siteRadiusSpin.setValue(3.0)
        self.siteRadiusSpin.setSingleStep(0.5)
        self.siteRadiusSpin.setDecimals(1)
        layout.addRow("Site margin:", self.siteRadiusSpin)

        self.alignMethodCombo = QComboBox()
        self.alignMethodCombo.addItems(list(map(str, AligMethod)))
        self.alignMethodCombo.setCurrentText(AligMethod.CEALIGN)
        layout.addRow("Align method:", self.alignMethodCombo)

        self.omegaSymbols = QLineEdit("*:.")
        layout.addRow("Clustal Omega", self.omegaSymbols)

        self.qualifierLine = QLineEdit(RMSF_DEFAULT_QUALIFIER)
        layout.addRow("Qualifier:", self.qualifierLine)

        self.prettyCheck = QCheckBox()
        layout.addRow("Pretty:", self.prettyCheck)

        calcButton = QPushButton("Calc")
        calcButton.clicked.connect(self.calc)
        layout.addWidget(calcButton)

    def calc(self):
        rmsf(
            prot_expr=self.proteinExprLine.text().strip(),
            ref_site=self.refSiteLine.text().strip(),
            site_radius=self.siteRadiusSpin.value(),
            align_method=self.alignMethodCombo.currentText(),
            omega_conservation=self.omegaSymbols.text().strip(),
            qualifier=self.qualifierLine.text().strip(),
            pretty=self.prettyCheck.isChecked(),
            quiet=False,
        )


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(300, 250)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle("(XDrugPy) Multi")

        tab = QTabWidget()
        tab.addTab(FechSimilarWidget(), "Finder")
        tab.addTab(RmsfWidget(), "RMSF")

        layout.addWidget(tab)


dialog = None


def run_plugin_gui():
    global dialog
    if dialog is None:
        dialog = MainDialog()
    dialog.show()


def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("(XDrugPy) Multi", run_plugin_gui)
