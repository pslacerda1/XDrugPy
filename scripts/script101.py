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

        max_num_cs=8,   # Read up to x consensus sites of the structure.
                        #   Try to increase and check if you get more hotspots.
                        #   However it may freeze the script as may exists too
                        #   many combinations to look for hotspots (combinatorial
                        #   explosion).
        min_cs_strength=5,  # Consensus sites with less than 5 probe clusters are
                            #   ignored.

        # Combinatory search.
        deep_search=True,   # Do combinatorial search. Unrelated to neural networks.
        remove_nested=True, # If a hotspot is subset of another, keep only the
                            #   bigger.

        # Steric clash detection algorithm.
        #   Clashes may turn a hotspot infeasible, if the clash is circunvented
        #   with the interaction mediated by another consensus site, this re-enables
        #   the hotspot.
        #
        num_pseudoatoms=25, # For any two atoms from two consensus sites in a
                            #   potential hotspot, 25 in-between points will
                            #   be checked for collision.
        clash_threshold=0.15,   # Tolerate up to 15% of collision accounting all
                                #   in-between points.
        pseudoatom_radius=1.5,  # The points are pseudo-atoms with 1.5 radii.
    )

    # now show me the results
    print(f"\n\n############# {my_label}")
    print("**** CLUSTERS ****")
    for cs in ftmap.clusters:
        pp(cs)

    print(f"\n\n############# {my_label}")
    print("**** HOTSPOTS ****")
    for hs in ftmap.hotspots:
        pp(object=hs)