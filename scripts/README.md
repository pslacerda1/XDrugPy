Welcome to the XDrugPy/scripts directory!
===

1. If you don't have a functional plugin yet, first follow the installation steps at https://tiny.cc/XDrugPy and get a working copy of the XDrugPy plugin.
2. Then, run the next command inside the PyMOL console to install XDrugPy also as a Python package.

        pip install -U https://github.com/pslacerda1/XDrugPy/archive/refs/heads/master.zip


3. Copy the script you want to a folder you own and edit and adapt the script to your needs. Ensure you're using a sane text editor (like Notepad++ or VSCode) to edit the file. You'll need to change filenames, PDB codes and options. Use asterisks as wildcards in filenames and globs!

4. Now, on the PyMOL command line, run your adapted script using the commands below. If needed, you may navigate and list folder contents with the `cd` and `ls` commands inside PyMOL prompt like a regular terminal. (I'm assuming you copied `hca.py` and adapted into `~/Documents/hca.py`.)
        
        cd ~/Documents
        run hca.py

or simply

        run ~/Documents/hca.py

 5. Just relax and your results will pop-up as intended.


 Good Luck!
 Pedro Lacerda