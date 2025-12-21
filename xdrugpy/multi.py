import time
import logging
from pymol import cmd as pm
from functools import lru_cache
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
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from strenum import StrEnum

from .utils import new_command, mpl_axis, PyMOLComboObjectBox, AligMethod


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
            except Exception as exc:
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
    pm.order(assembly_ids, True)  # XXX segfaults
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
    site_margin: float = 3.0,
    qualifier: str = RMSF_DEFAULT_QUALIFIER,
    pretty: bool = True,
    axis: str = "",
):
    """
    DESCRIPTION
        Calculate the RMSF of multiple related structures.

        A reference site must be supplied to focus, however full protein
        analysis can be achieved with a star * wildcard. A protein
        expression based on fnmatch to select the structures to calculate
        the RMSF must also be supplied.

    OPTIONS
        prot_expr: str
            An expression to select the structures to calculate the RMSF.
        ref_site: str
            A site expression to focus the RMSF calculation.
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
        f"(%{f0} & polymer) within {site_margin} of ({ref_site})",
        "site.add((resn,int(resi),chain))",
        space={"site": site},
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
            idx_resids = "+".join(str(r[1]) for r in site)  # if r[2] == chain)
            sele += f" i. {idx_resids}"
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
                X[resi] = np.empty((0, 3))
            X[resi] = np.vstack([X[resi], coords])

    # Sort residues
    X = {k: X[k] for k in sorted(X, key=lambda z: (z[2], z[1]))}

    # Find mean positions for each residue
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
        label = "%s %s:%s" % resi
        pm.alter(f"{f0} & i. {resi[1]} & c. {resi[2]}", f"p.rmsf={rmsf}")
        LABELS.append(label)
        RMSF.append(rmsf)

    # Show data
    if pretty:
        pm.hide("everything", f"{f0} & polymer")
        pm.show_as("line", f"{f0} & polymer")
        pm.spectrum("p.rmsf", "rainbow", f"{f0} & polymer")

    with mpl_axis(axis, constrained_layout=True) as ax:
        ax.bar(LABELS, RMSF)
        ax.set_ylabel("RMSF")
        ax.tick_params(axis="x", rotation=90)

    return RMSF, LABELS

    
@new_command
def pca(
    prots_sele: str,
    site_sele: str = '*',
    site_radius: float = 4.0,
    min_explained_variance: float = 0.9,
    axis: str = '',
):
    objects = []
    for obj in pm.get_object_list(f'({prots_sele}) & polymer'):
        objects.append(obj)
    assert len(objects) > 0, "Please review your selections"
    
    mappings = np.empty((0, 7))
    site = []
    
    # Take first object as reference
    for at in pm.get_model(f'{objects[0]} & ({site_sele}) & name CA').atom:
        site.append(at.index)
        mappings = np.vstack([
            mappings,
            (at.index, at.model, at.chain, at.resi, *at.coord)
        ])
    # Iter objects
    for obj in objects[1:]:
        try:
            # Align (without fit transform) objects into reference
            aln_obj = pm.get_unused_name()
            pm.extra_fit(
                f"({obj}) & polymer & name CA",
                f"({objects[0]}) & polymer & name CA",
                method="cealign",
                transform=0,
                object=aln_obj,
            )
            aln = pm.get_raw_alignment(aln_obj)
        finally:
            pm.delete(aln_obj)
        
        # Map bijections from object to reference
        for (obj1, idx1), (obj2, idx2) in aln:
            if idx1 not in site:
                continue
            for at in pm.get_model(f"%{obj2} & index {idx2} & name CA").atom:
                mappings = np.vstack([
                    mappings,
                    (idx1, at.model, at.chain, at.resi, *at.coord)
                ])
    
    dtypes = {
        "index": int,
        "model": str,
        "chain": str,
        "resi": int,
        "x": float,
        "y": float,
        "z": float
    }
    mappings = pd.DataFrame(mappings, columns=list(dtypes)).astype(dtypes)
    
    rprint("\n**Principal Component Analisys**\n")

    rprint(f"Num of models: {len(mappings['model'].unique())}")
    rprint(f"Total mapped residues: {len(site)}")
    rprint()

    table = Table(title="Num of Cα per model")
    table.add_column("Model", justify="right")
    table.add_column("Num of Cα", justify="center")
    index_counts = mappings.groupby('model')['index'].nunique()
    for model, num_ca in index_counts.items():
        table.add_row(model, str(num_ca))

    console = Console()
    console.print(table)
    
    # Extract C-alpha XYZ vector of full mapped sequences
    vectors = []
    for model, group in mappings.sort_values('index').groupby('model'):
        vectors.append(
            group
            .set_index("index")
            .reindex(site)
            [['x', 'y', 'z']]
            .to_numpy()
            .ravel()
        )
    
    # Preprocessing
    X = np.array(vectors)
    print(X)
    missing_ca = np.isnan(X[:,1]).sum() / len(X[:,1])
    rprint(f"\nPercent of unmatched Cα: {missing_ca:.2f}%")

    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    
    n_pc = np.argmax(cum >= min_explained_variance)+1
    rprint(f"Num of PCs to explain {min_explained_variance*100:.2f}% of variance: {n_pc}")
    rprint()

    pca = PCA(n_components=n_pc)
    X_pca = pca.fit_transform(X)
    
    opt = OPTICS(cluster_method='xi')
    labels = opt.fit_predict(X_pca)

    uniq_labels = list(sorted(set(labels)))
    cmap = plt.cm.nipy_spectral
    colors = cmap(np.linspace(0, 1, len(uniq_labels)))
    cmap = ListedColormap(colors)
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, constrained_layout=True)

    if X_pca.shape[1] >= 2:
        ax0.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap)
        ax0.set_xlabel("PC1")
        ax0.set_ylabel("PC2")
    else:
        ax0.text(s="Not enought PCs.", ha="center")

    if X_pca.shape[1] >= 3:
        ax1.scatter(X_pca[:, 0], X_pca[:, 2], c=labels, cmap=cmap)
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC3")
    else:
        ax1.text(s="Not enought PCs.", ha="center")

    if X_pca.shape[1] >= 3:
        ax2.scatter(X_pca[:, 1], X_pca[:, 2], c=labels, cmap=cmap)
        ax2.set_xlabel("PC2")
        ax2.set_ylabel("PC3")
    else:
        ax2.text(s="Not enought PCs.", ha="center")
    
    handles = []
    for value, color in zip(uniq_labels, cmap.colors):
        handles.append(plt.Line2D(
            [], [], marker="o", linestyle="", color=color,
            label=str(value)
        ))

    plt.legend(handles=handles, title="Cluster")
    
    ax3.plot(range(1, len(evr)+1), cum)
    ax3.axhline(y=0.9, c='red', ls='--')
    ax3.set_xlabel("Number of PCs")
    ax3.set_ylabel("Cumulative Explained Variance")


    if axis and isinstance(axis, (str)):
        fig.savefig(axis)
    else:
        fig.show()


# Uniprot = namedtuple(
#     'Uniprot',
#     'pdb_id uniprot_id chain_id uniprot_name length'
# )
# @cache
# def get_uniprot_from_pdbe(pdb_id: str):
#     """
#     Retrieves UniProt ID(s) for a given PDB ID using the PDBe API.
#     """
#     session = requests.Session()
#     pdb_id = pdb_id.lower()
#     url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
#     response = session.get(url)
#     data = response.json()
#     ret = []
#     if pdb_id in data and "UniProt" in data[pdb_id]:
#         done = set()
#         for uniprot_id, mappings in data[pdb_id]['UniProt'].items():
#             uniprot_name = mappings['name']
#             for mapping in mappings['mappings']:
#                 start = mapping['unp_start']
#                 end = mapping['unp_end']
#                 length = end - start + 1
#                 chain_id = mapping['chain_id']
#                 entry =  Uniprot(pdb_id, uniprot_id, chain_id, uniprot_name, length)
#                 if entry in done:
#                     continue
#                 done.add(entry)
#                 ret.append(entry)
#     return ret


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

        self.siteMarginSpin = QDoubleSpinBox()
        self.siteMarginSpin.setMinimum(1.0)
        self.siteMarginSpin.setMaximum(10.0)
        self.siteMarginSpin.setValue(3.0)
        self.siteMarginSpin.setSingleStep(0.5)
        self.siteMarginSpin.setDecimals(1)
        layout.addRow("Site margin:", self.siteMarginSpin)

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
            site_margin=self.siteMarginSpin.value(),
            qualifier=self.qualifierLine.text().strip(),
            pretty=self.prettyCheck.isChecked(),
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
