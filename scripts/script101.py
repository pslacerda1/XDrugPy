#
# Hello newbie!
#
# To run this file you may need to adapt the lines 11 to 15.
# In case of doubts ask GPT or send a question...
#
from pprint import pp
from xdrugpy import load_ftmap

# folder with the pdb files...
pkg_data = "./tests/data"
for pdb in ['1dqa', '1dq8', '1dq9']:  # ... pdb files inside the folder
    
    # the full name of the pdb file
    my_file = f"{pkg_data}/{pdb}_atlas.pdb"   # looks like "./tests/data/1dq8_atlas.pdb"
    
    # a friendly label for you PyMOL object
    my_label = pdb
    
    # drill-down inside load_ftmap docstring to know more
    ftmap = load_ftmap(
        filename=my_file,
        group=my_label,

        ### advanced options below ###

        # combinatory search
        deep_search=True,
        max_size=8, # try to increase and check if you get more hotspots,
                    #   but it may freeze the script as may exists too
                    #   many combinations
        remove_nested=True,

        # clash algorithm
        clash_threshold=0.15,
        num_pseudoatoms=25,
        pseudoatom_radius=0.5
    )

    # now show me the results
    print(f"\n\n############# {my_label}")
    print("**** CLUSTERS ****")
    for cs in ftmap.clusters:
        pp(cs)

    print(f"\n\n############# {my_label}")
    print("\n**** HOTSPOTS ****")
    for hs in ftmap.hotspots:
        pp(hs)
