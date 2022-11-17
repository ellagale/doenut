############################################################################################################
#
#              DoENUT Designer
#
############################################################################################################

# !!! TO-DO !!!
#
# make this into a nice proper class  

import pandas as pd

# this is where the experiments come from
from doepy.build import full_fact
from doepy.build import frac_fact_res

####### this should be a base class with hte actual experiment overwritten in the sub-class
def experiment_designer(levels, 
                        res,
                        do_midpoints=True,
                        shuffle=True,
                        repeats=1,
                        num_midpoints=3):
    """
    levels is a dictionary of factor name and levels
    res is the resolution (for frac fact) - shouldn't be in class
    do_midpoints whether to add in the mid points
    shuffle whether to shuffle
    repeats how many repeats you're doing of the NON-MIDPOINTS
    num_midpoints, how many midpoints to do
    """

    # deepcopy as their code overwrites the levels >:(
    levels_in = copy.deepcopy(levels)
    design = frac_fact_res(levels_in, res=res)
    factor_names = [x for x in levels.keys()]
    if repeats > 1:
        for i in range(repeats):
            design = design.append(midpoints, ignore_index=True)
    if do_midpoints:
        midpoints = {}
        for factor in levels.keys():
            if len(levels[factor]) > 2:
                midpoints[factor] = np.median(levels[factor])
            else:
                midpoints[factor] = np.mean(levels[factor])
        #midpoints = pd.DataFrame(midpoints, index=str(len(design)+1))
        for i in range(num_midpoints):
            design = design.append(midpoints, ignore_index=True)
        
        if shuffle == True:
            design = design.sample(frac=1)
                
    return design



def frac_fact_res_designer(levels, 
                         res,
                        do_midpoints=True,
                        shuffle=True,
                        repeats=1,
                        num_midpoints=3):

    levels_in = copy.deepcopy(levels)
    design = frac_fact_res(levels_in, res=res)
    factor_names = [x for x in levels.keys()]
    if do_midpoints:
        midpoints = {}
        for factor in levels.keys():
            if len(levels[factor]) > 2:
                midpoints[factor] = np.median(levels[factor])
            else:
                midpoints[factor] = np.mean(levels[factor])
        #midpoints = pd.DataFrame(midpoints, index=str(len(design)+1))
        for i in range(num_midpoints):
            design = design.append(midpoints, ignore_index=True)
        
        if shuffle == True:
            design = design.sample(frac=1)
                
    return design
