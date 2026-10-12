from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from subprocess import CalledProcessError, check_call
from urllib.request import Request, urlopen

import requests
from pymol import Qt

from .paths import RESOURCES_DIR, VERSION_FILE

RUST_PROGRAM_VERSION = "v.40"



@dataclass
class Commit:
    sha: str
    datetime: str
    message: str

    @property
    def abbrev_sha(self):
        return self.sha[:7]

    @property
    def title(self):
        return self.message.split('\n')[0]

    @property
    def date(self):
        return self.datetime[:10]
    
    def __str__(self):
        return f"{self.date}: {self.abbrev_sha} - {self.title}"

    @classmethod
    def recent_from_github(cls, timeout=5.0, since=None, until=None) -> list[Commit]:
        params = {
            "per_page": 100
        }
        if since:
            params['since'] = f"{since}T00:00:00Z"
        if until:
            params['until'] = f"{until}T23:59:59Z"
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "XDrugPy-PyMOL-Plugin",
        }
        commit_ls: list[Commit] = []
        try:
            recent_commits_url = f"https://api.github.com/repos/pslacerda1/XDrugPy/commits"
            data = requests.get(
                recent_commits_url,
                params=params,
                headers=headers,
                timeout=timeout
            ).json()

            for json_commit in data:
                sha = json_commit['sha']
                commit = Commit(
                    sha=sha,
                    datetime=json_commit['commit']['committer']['date'],
                    message=json_commit['commit']['message']
                )
                commit_ls.append(commit)
        except TimeoutError:
            pass
        return commit_ls

    @classmethod
    def from_github(cls, sha, timeout=5.0) -> Commit:
        commit_url = f"https://api.github.com/repos/pslacerda1/XDrugPy/commits/{sha}"
        json_commit = requests.get(
            commit_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "XDrugPy-PyMOL-Plugin",
            },
            timeout=timeout
        ).json()
        assert sha == json_commit['sha']
        return Commit(
            sha=sha,
            datetime=json_commit['committer']['date'],
            message=json_commit['message']
        )


def _retrieve_github_url(url) -> bytes:
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        return resp.read()


def install_plugin(plugin_version):

    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    #
    # Record a version file
    #
    commit = Commit.from_github(plugin_version)
    VERSION_FILE.write_text(f'{commit.sha}\n{commit.datetime}\n{commit.title}')
    
    #
    # Install pip dependencies
    #
    try:
        #
        # The plugin itself
        check_call([
            sys.executable, "-m", "pip", "install",
            f"https://github.com/pslacerda1/XDrugPy/archive/{version_sha}.zip",
        ])
        check_call([
            sys.executable, "-m", "pip", "install",
            "-r", f"http://raw.githubusercontent.com/pslacerda1/XDrugPy/{version_sha}/requirements.txt"
        ])

        #
        # Override requirements versions
        #   These are very old versions not installable in requirements.txt
        #   because of conflicts.
        check_call([
            sys.executable, "-m", "pip", "install", "numpy==1.26.4", "scipy==1.15.3"
        ])

        #
        # A specific feature I want into PyMOL
        check_call([
            sys.executable, "-m", "pip", "install", "--no-deps",
            "https://github.com/pslacerda1/pymol_new_command/archive/refs/heads/main.zip"
        ])

        #
        # The --no-deps option is global
        #   Not applicable per-package in requirements.txt
        try:
            check_call([
                sys.executable, "-m", "pip", "install", "--no-deps",
                "pyKVFinder==0.9.5",
            ])
        except CalledProcessError as exc:
            # Probably on an old MacOs
            print("Continuing without pyKVFinder.")

    except CalledProcessError as exc:
        raise SystemError(f"XDrugPy: Installation failed.") from exc

    #
    # Install the low-level Rust XDrugPy tool
    system = platform.system().lower()
    match system:
        case "linux":
            web_name = "xdrugpy_xhf-ubuntu"
        case "windows":
            web_name = "xdrugpy_xhf-windows.exe"
        case "darwin":
            web_name = "xdrugpy_xhf-macos"
        case _:
            raise RuntimeError("Unexpected system.")
    url = f"https://github.com/pslacerda1/xdrugpy_xhf/releases/download/{RUST_PROGRAM_VERSION}/{web_name}"
    exe = RESOURCES_DIR / "xdrugpy_xhf"
    if system == "windows":
        exe = exe.with_suffix('.exe')
    exe.unlink(missing_ok=True)
    exe.write_bytes(_retrieve_github_url(url))
    os.chmod(exe, stat.S_IXUSR)

    #
    # Install Vina
    #   Downloading binaries
    match system:
        case "windows":
            web_name = "vina_1.2.7_win.exe"
        case "linux":
            web_name = "vina_1.2.7_linux_x86_64"
        case "darwin":
            web_name = "vina_1.2.7_mac_x86_64"
        case _:
            raise RuntimeError("Unexpected system.")

    url = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{web_name}"
    exe = RESOURCES_DIR / 'vina'
    if system == "windows":
        exe = exe.with_suffix('.exe')
    exe.unlink(missing_ok=True)
    exe.write_bytes(_retrieve_github_url(url))
    os.chmod(exe, stat.S_IXUSR)

    #
    # Install Clustal Omega
    #   As a conda package (or downloading and unpacking the zipfile on Windows)
    match system:
        case "windows":
            local_zip = RESOURCES_DIR / web_name
            local_exe = RESOURCES_DIR / 'clustalo.exe'
            local_zip.unlink(missing_ok=True)
            local_exe.unlink(missing_ok=True)
            
            url = f"https://github.com/pslacerda1/XDrugPy/raw/refs/heads/master/misc/clustal-omega-1.2.2-win64.zip"
            local_zip.write_bytes(_retrieve_github_url(url))

            zipfile.ZipFile(local_zip).extractall(RESOURCES_DIR)
            for file in (RESOURCES_DIR / "clustal-omega-1.2.2-win64").glob("*"):
                shutil.move(file, RESOURCES_DIR)
            os.chmod(local_exe, stat.S_IXUSR)

        case "darwin" | "linux":
            check_call([
                'conda', 'install', '-y', 'bioconda::clustalo'
            ])



QWidget = Qt.QtWidgets.QWidget
QFormLayout = Qt.QtWidgets.QFormLayout
QTextEdit = Qt.QtWidgets.QTextEdit

# QPushButton = Qt.QtWidgets.QPushButton
# QSpinBox = Qt.QtWidgets.QSpinBox
# QDoubleSpinBox = Qt.QtWidgets.QDoubleSpinBox
# QLineEdit = Qt.QtWidgets.QLineEdit
# QCheckBox = Qt.QtWidgets.QCheckBox
# QVBoxLayout = Qt.QtWidgets.QVBoxLayout
# QHBoxLayout = Qt.QtWidgets.QHBoxLayout
QDialog = Qt.QtWidgets.QDialog
QComboBox = Qt.QtWidgets.QComboBox
# QTabWidget = Qt.QtWidgets.QTabWidget
QLabel = Qt.QtWidgets.QLabel
# QTableWidget = Qt.QtWidgets.QTableWidget
# QTableWidgetItem = Qt.QtWidgets.QTableWidgetItem
# QGroupBox = Qt.QtWidgets.QGroupBox
QHeaderView = Qt.QtWidgets.QHeaderView
QTreeWidget = Qt.QtWidgets.QTreeWidget
QTreeWidgetItem = Qt.QtWidgets.QTreeWidgetItem

QtCore = Qt.QtCore
QIcon = Qt.QtGui.QIcon

class GithubShaComboBox(QComboBox):

    def __init__(self):
        super().__init__()
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setEditText("")
        self.recent_commits: list[Commit] = []
        
        # initial seed
        self.currentData()

    def showPopup(self):
        currentText = self.currentText().strip()
        self.clear()
        for commit in self.recent_commits:
            self.addItem(str(commit), commit)
            
        if currentText != "":
            self.setCurrentText(currentText)
        super().showPopup()

    def currentText(self) -> str:
        return super().currentText().strip()
    
    def currentData(self):
        if commit := super().currentData():
            return commit
    
        text = self.currentText()
        re_date = r'(?P<Y>\d{4})-(?P<M>\d{2})(-(?P<D>\d{2}))?'
        re_sha = r'[a-z0-9]{7,40}'

        if match := re.match(re_date, text):
            yyyy: str = match.group('Y')
            mm: str = match.group('mm')
            dd: str | None = match.group('dd')
            commits = Commit.recent_from_github(
                since=f'{yyyy}-{mm}-{dd}',
                until=f'{yyyy}-{mm}-{dd}',
            )
            self.recent_commits = commits
        elif re.match(re_sha, text):
            self.recent_commits = [
                c for c in Commit.recent_from_github()
                if c.sha.startswith(text)
            ]
        else:
            self.recent_commits = Commit.recent_from_github()
        return None

class UpdaterDialog(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("(XDrugPy) Settings")

        self.form = QFormLayout()
        self.setLayout(self.form)

        self.text_area = QTextEdit()
        try:
            self.text_area.setPlainText(self.get_version_string())
        except FileNotFoundError:
            pass
        self.text_area.setReadOnly(True)

        self.commit_combo = GithubShaComboBox()
        self.form.addRow("Recent commits:", self.commit_combo)

    def get_version_string(self):
        try:
            version_sha, version_date , version_title= VERSION_FILE.read_text().strip().splitlines()
            return f"""
                      <b><i>XDrugPy</i></b>
                <br/> <b>Cite the old DOI:</b>  <a>https://doi.org/10.1007/s10822-021-00403-8</a>
                <br/> <b>Github commit:</b>  {version_sha}
                <br/> <b>Commit date:</b>  {version_date}
                <br/> <b>Commit title:</b>  {version_title}
            """
        except FileNotFoundError:
            return """
                <b><i>XDrugPy</i></b>
                <br/> <b>Cite the old DOI:</b>  <a>https://doi.org/10.1007/s10822-021-00403-8</a>
                <font color=red><b>Ready to install!</b></font>
            """
        except Exception:
            return """
                <b><i>XDrugPy</i></b>
                <br/> <b>Cite the old DOI:</b> <a>https://doi.org/10.1007/s10822-021-00403-8</a>
                <font color=red><b>Need to reinstall, choose a commit.</b></font>
            """