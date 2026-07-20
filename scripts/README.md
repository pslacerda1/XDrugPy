## Welcome to the XDrugPy/scripts directory!

1. If you don't have a functional plugin yet, first follow the installation steps at https://tiny.cc/XDrugPy and get a working copy of the ***XDrugPy*** plugin.
2. Then, run the next command inside the PyMOL console (yes, *inside*) to install XDrugPy also as a Python package.
```bash
        pip install -U https://github.com/pslacerda1/XDrugPy/archive/refs/heads/master.zip
```

3. Copy the script you want to a folder you own and edit and adapt the contents to your needs. Ensure you're using a sane text editor (like Notepad++ or VSCode) to edit the file. You'll need to change filenames, PDB codes and options. Use asterisks and question marks as wildcards in filenames and globs!

**[Optional]** If you need sample FTMap/FTMove (or Atlas) files to run your adapted scripts, download these from [this repository](https://github.com/pslacerda1/XDrugPy/tree/master/tests/data). After downloading the raw file (or copy and paste the contents) into a local new file, open them with the text editor to examine the PDB contents, then close it. It should work for you.

    * XDrugPy/tests/data/1dq8_atlas.pdb
    * XDrugPy/tests/data/1dq9_atlas.pdb
    * XDrugPy/tests/data/1dqa_atlas.pdb

4. Now, on the PyMOL command line, run your adapted script using the commands below. If needed, you may navigate and list folder contents with the `cd` and `ls` commands inside PyMOL prompt like a regular terminal. (I'm assuming you copied `script101.py` and adapted into `~/Documents/script101.py`.)
```bash
        cd ~/Documents
        run script101.py

        # or simply

        run ~/Documents/script101.py
```
 
Now you just relax and your results will pop-up as intended in a while.

Good Luck!
 
Pedro Lacerda
