############################################################################################################
#
#              DoENUT Designer
#
############################################################################################################

# !!! TO-DO !!!
#
# make this into a nice proper class
import pandas as pd
import numpy as np
import copy

# this is where the experiments come from
from doepy.build import frac_fact_res


def full_fact(**kwargs):
    """
    Generate a full factorial model from the supplied parameters
    :param kwargs: keyworded lists of allowed values for each parameter
    :return: a panda datafram of the all the experiments
    """
    # first validate the inputs are all lists or list like
    # while we are here, work out how bit this is.
    row_count = 1
    for key, value in kwargs.items():
        try:
            _ = iter(value)
        except TypeError as e:
            print(f"Parameter {key} is not iterable")
            raise e
        row_count = row_count * len(value)
    result = np.zeros((row_count, len(kwargs.keys())), dtype="O")

    # Now build up the data column by column
    # how many row 'blocks' there are to the left of the current column
    left_data = 1
    # how many rows are remaining to fill.
    right_data = row_count

    # Note, there are a lot of int() calls below,
    # but these should always be valid as we are dividing ints by other ints
    # that are divisors of it.
    for column_idx, (column, values) in enumerate(kwargs.items()):
        value_count = len(values)
        # how many times do we need to write this value in a row?
        rows_per_value = int(right_data / value_count)
        for group in range(left_data):
            # work out where this group begins
            offset = rows_per_value * value_count * group
            for idx, value in enumerate(values):
                start = int(offset + (idx * rows_per_value))
                end = int(start + rows_per_value)
                result[start:end, column_idx] = value
        # re-establish the invariants
        left_data = int(left_data * value_count)
        right_data = right_data / value_count

    result = pd.DataFrame(
        columns=list(kwargs.keys()), dtype=object, data=result
    )
    return result


# TODO:: this should be a base class with hte actual experiment overwritten in the sub-class
def experiment_designer(
    levels, res, do_midpoints=True, shuffle=True, repeats=1, num_midpoints=3
):
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
        # midpoints = pd.DataFrame(midpoints, index=str(len(design)+1))
        for i in range(num_midpoints):
            design = design.append(midpoints, ignore_index=True)

        if shuffle:
            design = design.sample(frac=1)

    return design


def frac_fact_res_designer(
    levels, res, do_midpoints=True, shuffle=True, repeats=1, num_midpoints=3
):
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
        # midpoints = pd.DataFrame(midpoints, index=str(len(design)+1))
        for i in range(num_midpoints):
            design = design.append(midpoints, ignore_index=True)

        if shuffle:
            design = design.sample(frac=1)

    return design


def fact_designer(
    levels, do_midpoints=True, shuffle=True, repeats=1, num_midpoints=3
):
    levels_in = copy.deepcopy(levels)
    design = full_fact(levels_in)
    factor_names = [x for x in levels.keys()]
    if do_midpoints:
        midpoints = {}
        for factor in levels.keys():
            if len(levels[factor]) > 2:
                midpoints[factor] = np.median(levels[factor])
            else:
                midpoints[factor] = np.mean(levels[factor])
        # midpoints = pd.DataFrame(midpoints, index=str(len(design)+1))
        for i in range(num_midpoints):
            design = design.append(midpoints, ignore_index=True)

        if shuffle:
            design = design.sample(frac=1)

    return design
