import sys
import json
import os
import platform
import shutil
import stat
import zipfile
import urllib.request
from urllib.request import urlretrieve
from subprocess import check_call, CalledProcessError

from . import VERSION_FILE, RESOURCES_DIR


RUST_PROGRAM_VERSION = "v.40"


def install_plugin(plugin_version):


    #
    # Record a version file
    #
    github_repo_url = f"https://api.github.com/repos/pslacerda1/XDrugPy/commits/{plugin_version}"
    with urllib.request.urlopen(github_repo_url) as response:
        text = response.read().decode("utf-8")
    
    data = json.loads(text)
    version_sha = data['sha']
    version_date = data['commit']['committer']['date']

    
    VERSION_FILE.write_text(version_sha + '\n' + version_date)
    
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
    if exe.exists():
        os.unlink(exe)
    urlretrieve(url, exe)
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
    if exe.exists():
        os.unlink(exe)
    urlretrieve(url, exe)
    os.chmod(exe, stat.S_IXUSR)

    #
    # Install Clustal Omega
    #   As a conda package (or downloading and unpacking the zipfile on Windows)
    match system:
        case "windows":
            web_name = "clustal-omega-1.2.2-win64.zip"
            local_zip = RESOURCES_DIR / web_name
            local_exe = RESOURCES_DIR / 'clustalo.exe'
            if local_zip.exists():
                os.unlink(local_zip)
            urlretrieve(
                f"https://github.com/pslacerda1/XDrugPy/raw/refs/heads/master/misc/{web_name}",
                local_zip
            )
            zipfile.ZipFile(local_zip).extractall(RESOURCES_DIR)
            for file in (RESOURCES_DIR / "clustal-omega-1.2.2-win64").glob("*"):
                shutil.move(file, RESOURCES_DIR)
            os.chmod(local_exe, stat.S_IXUSR)

        case "darwin" | "linux":
            check_call([
                'conda', 'install', '-y', 'bioconda::clustalo'
            ])
