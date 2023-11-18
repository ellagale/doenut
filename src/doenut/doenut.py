"""
DOENUT
Design of Experiments Numerical Utility Toolkit
"""

# first we import some useful libraries
import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
from typing import Tuple
from doenut.data import ModifiableDataSet
from doenut.models import AveragedModel


def orthogonal_scaling(inputs, axis=0):
    # the scaling thingy that Modde uses
    inputs_max = np.max(inputs, axis)
    inputs_min = np.min(inputs, axis)
    Mj = (inputs_min + inputs_max) / 2
    Rj = (inputs_max - inputs_min) / 2
    scaled_inputs = (inputs - Mj) / Rj
    return scaled_inputs, Mj, Rj


def scale_1D_data(scaler, data, do_fit=True):
    if do_fit:
        scaler.fit(data.reshape(-1, 1))
    data_scaled = scaler.transform(data.reshape(-1, 1))
    data_scaled = data_scaled.reshape(-1)
    return data_scaled, scaler


def scale_by(new_data, Mj, Rj):
    # the scaling thingy that Modde uses
    # TODO:: Any form of sanity checking whatsoever.
    new_data = (new_data - Mj) / Rj
    return new_data


def find_replicates(inputs):
    """Find experimental settings that are replicates"""
    # list comps ftw!
    a = [x for x in inputs[inputs.duplicated()].index]
    b = [x for x in inputs[inputs.duplicated(keep="last")].index]
    replicate_row_list = np.unique(np.array(a + b))
    return replicate_row_list


def train_model(
    inputs,
    responses,
    test_responses,
    do_scaling_here=False,
    fit_intercept=False,
    verbose=True,
):
    """A simple function to train a model
    :param inputs: full set of terms for the model (x_n)
    :param responses: expected responses for the inputs (ground truth, y)
    :param test_responses: expected responses for seperate test data (if used)
    :param do_scaling_here: whether to scale the data
    :param fit_intercept: whether to fit the intercept
    :param verbose: whether to perform additional logging.
    :return: A tuple of:
        A model fitted to the data,
        the inputs used
        the R2 of that model
        the predictions that model makes for the original inputs
    """
    if do_scaling_here:
        inputs, _, _ = orthogonal_scaling(inputs, axis=0)
    if test_responses is None:
        test_responses = responses
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(inputs, responses)
    predictions = model.predict(inputs)
    R2 = model.score(inputs, test_responses)
    if verbose:
        print("R squared for this model is {:.3}".format(R2))
    return model, inputs, R2, predictions


def Calculate_R2(ground_truth, predictions, key, word="test", verbose=True):
    """Calculates R2 from input data
    You can use this to calculate q2 if you're
    using the test ground truth as the mean
    else use calculate Q2
    I think this is what Modde uses for PLS fitting"""
    errors = ground_truth[[key]] - predictions[[key]]
    test_mean = np.mean(ground_truth[[key]], axis=0)
    if verbose:
        print(f"Mean of {word} set: {test_mean[0]}")
    errors["squared"] = errors[key] * errors[key]
    sum_squares_residuals = sum(errors["squared"])
    if verbose:
        print(
            "Sum of squares of the residuals (explained variance) is"
            f"{sum_squares_residuals}"
        )
    sum_squares_total = sum((ground_truth[key] - test_mean[0]) ** 2)
    if verbose:
        print(f"Sum of squares total (total variance) is {sum_squares_total}")
    R2 = 1 - (sum_squares_residuals / sum_squares_total)
    if word == "test":
        print("{} is {:.3}".format("Q2", R2))
    else:
        print("{} is {:.3}".format("R2", R2))
    return R2


def Calculate_Q2(
    ground_truth, predictions, train_responses, key, word="test", verbose=True
):
    """A different way of calculating Q2
    this uses the mean from the training data, not the
    test ground truth"""
    errors = ground_truth[[key]] - predictions[[key]]
    train_mean = np.mean(train_responses[[key]], axis=0)
    test_mean = np.mean(ground_truth[[key]], axis=0)
    if verbose:
        print(f"Mean of {word} set: {test_mean.iloc[0]}")
        print(f"Mean being used: {train_mean.iloc[0]}")
    errors["squared"] = errors[key] * errors[key]
    sum_squares_residuals = sum(errors["squared"])
    if verbose:
        print(
            "Sum of squares of the residuals (explained variance) is"
            f"{sum_squares_residuals}"
        )
    sum_squares_total = sum((ground_truth[key] - train_mean.iloc[0]) ** 2)
    # stuff from Modde
    # errors/1

    if verbose:
        print(f"Sum of squares total (total variance) is {sum_squares_total}")
    R2 = 1 - (sum_squares_residuals / sum_squares_total)
    if word == "test":
        print("{} is {:.3}".format("Q2", R2))
    else:
        print("{} is {:.3}".format("R2", R2))
    return R2


def dunk(setting=None):
    if setting == "coffee":
        print("Splash!")
    else:
        print("Requires coffee")
    return


def average_replicates(
    inputs: pd.DataFrame, responses: pd.DataFrame, verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """averages inputs that are the same
    TO-DO - you can make it pick nearly teh same inputs if
    if you add the actual values which are not always the expected values
    inputs = inputs
    responses = responses"""
    whole_inputs = inputs
    averaged_responses = pd.DataFrame()
    averaged_inputs = pd.DataFrame()

    duplicates = [x for x in whole_inputs[whole_inputs.duplicated()].index]
    duplicates_for_averaging = {}
    non_duplicate_list = [x for x in whole_inputs.index if x not in duplicates]
    for non_duplicate in non_duplicate_list:
        this_duplicate_list = []
        non_duplicate_row = whole_inputs.loc[[non_duplicate]]
        for duplicate in duplicates:
            duplicate_row = whole_inputs.loc[[duplicate]]
            if non_duplicate_row.equals(duplicate_row):
                this_duplicate_list.append(duplicate)
                if verbose:
                    print(
                        f"found duplicate pairs: {non_duplicate}, {duplicate}"
                    )
        if len(this_duplicate_list) > 0:
            duplicates_for_averaging[non_duplicate] = this_duplicate_list
        else:
            averaged_inputs = pd.concat([averaged_inputs, non_duplicate_row])
            averaged_responses = pd.concat(
                [averaged_responses, responses.iloc[[non_duplicate]]]
            )

    for non_duplicate, duplicates in duplicates_for_averaging.items():
        # print(f"nd: {non_duplicate}")
        to_average = whole_inputs.loc[[non_duplicate]]
        to_average_responses = responses.loc[[non_duplicate]]
        for duplicate in duplicates:
            to_average = pd.concat([to_average, whole_inputs.loc[[duplicate]]])
            to_average_responses = pd.concat(
                [to_average_responses, responses.loc[[duplicate]]]
            )
        meaned = to_average.mean(axis=0)
        meaned_responses = to_average_responses.mean(axis=0)
        try:
            averaged_inputs = pd.concat(
                [averaged_inputs, pd.DataFrame(meaned).transpose()],
                ignore_index=True,
            )
            averaged_responses = pd.concat(
                [
                    averaged_responses,
                    pd.DataFrame(meaned_responses).transpose(),
                ],
                ignore_index=True,
            )
        except TypeError:
            averaged_inputs = pd.DataFrame(meaned).transpose()
            averaged_responses = pd.DataFrame(meaned_responses).transpose()

    return averaged_inputs, averaged_responses


def calc_ave_coeffs_and_errors(coeffs, labels, errors="std", normalise=False):
    """Coefficient plot
    set error to 'std' for standard deviation
    set error to 'p95' for 95th percentile (
    approximated by 2*std)"""

    ave_coeffs = np.mean(coeffs, axis=0)[0]
    stds = np.std(coeffs, axis=0)[0]
    if normalise:
        ave_coeffs = ave_coeffs / stds
        stds = np.ones_like(ave_coeffs)
    if errors == "std":
        error_bars = stds
    elif errors == "p95":
        # this is an approximation assuming a gaussian distribution in your coeffs
        error_bars = 2 * stds
    else:
        raise ValueError(
            f"Error: errors setting {errors} not known, chose std or p95"
        )

    return ave_coeffs, error_bars


def autotune_model(
    inputs,
    responses,
    source_list,
    response_selector=[0],
    use_scaled_inputs=True,
    do_scaling_here=True,
    drop_duplicates="average",
    errors="p95",
    normalise=True,
    do_hierarchical=True,
    remove_significant=False,
    verbose=False,
):
    """
    inputs: the input matrix
    responses: the results
    response_selector=[0]: which column of results to use, or all of it
    use_scaled_inputs=True: scale model to remove columns baseed on stds
    do_scaling_here=True: if you want scaled inputs and haven't input them
    errors='p95': 95th percentile or 'std' for standard deviation
    normalise=True: setting for coefficient calculation - wants to match scaled inputs
    remove_significant: model will continue removing terms until only one is left
        verbose=False
    """
    sat_inputs = inputs

    if verbose:
        print(f"Source list is {source_list}")
    input_selector = [i for i in range(len(sat_inputs.columns))]
    output_indices = input_selector
    # global list of all column names.
    input_terms = list(sat_inputs.columns)
    output_terms = input_terms
    if verbose:
        print("numbers\tnames")
        for i, v in enumerate(input_terms):
            print(f"{i}\t{v}")
    this_model = None
    have_removed = True
    R2_over_opt = []
    Q2_over_opt = []
    n_terms_over_opt = []
    terms = []

    while have_removed:
        print("Beginning loop")
        selected_input_indices = output_indices
        selected_input_terms = output_terms
        if len(selected_input_indices) == 0:
            break
        data = ModifiableDataSet(sat_inputs, responses)
        if selected_input_terms:
            data.filter(selected_input_terms)
        this_model = AveragedModel(
            data, scale_run_data=True, drop_duplicates=drop_duplicates
        )

        selected_inputs = sat_inputs.iloc[:, selected_input_indices]
        selected_input_terms = list(selected_inputs.columns)
        R2_over_opt.append(this_model.r2)
        Q2_over_opt.append(this_model.q2)

        # cell 2
        # print("Cell 2:")
        print(f"Selected terms {selected_input_terms}")
        print(f"Source List: {source_list}")
        # build a dictionary mapping from input term to the
        # set of derived term's indices.
        # Note that we are only caring about indices still in
        # selected_input_indices (I.e. ones we have not thrown out!)
        dependency_dict = {}
        for i in selected_input_indices:
            dependency_dict[i] = set()
            # ignore 1st order terms. They have no antecedents
            if isinstance(source_list[i], str):
                continue
            # single term - a direct power of a 1st order term.
            if isinstance(source_list[i], int):
                if i in selected_input_indices:
                    dependency_dict[source_list[i]].add(i)
            # some other 2nd+ term.
            if isinstance(source_list[i], list):
                for x in source_list[i]:
                    if i in selected_input_indices:
                        try:
                            dependency_dict[x].add(i)
                        except:  # TODO:: fix blank except. I _think_ KeyError
                            if do_hierarchical:
                                print(
                                    "Error: Heirarchical model missing lower level terms!!!!"
                                )
        print(dependency_dict)
        # Handy shortcut - since the empty set is considered false,
        # we can just test dependency_dict[some_term] to see if there
        # are still dependents.

        # cell 3
        # enforce_hierarchical_model = do_hierarchical
        # whether to enforce hierarchy over terms (i.e. a lower order term
        # must be present for a higher order one)

        ave_coeffs, error_bars = calc_ave_coeffs_and_errors(
            coeffs=this_model.coeffs,
            labels=selected_input_terms,
            errors="p95",
            normalise=True,
        )
        n_terms_over_opt.append(len(ave_coeffs))

        # create a copy of source_list that we will modify for the next iteration
        output_indices = list(selected_input_indices)

        # build a list of all terms whose error bar crosses 0.
        # Values are tuples of the form (source_list index, |error_bar_size|)
        insignificant_terms = []
        for i in range(len(ave_coeffs)):
            if abs(ave_coeffs[i]) < abs(error_bars[i]):
                # print("{:.2}- {:.2}", ave_coeffs[i], error_bars[i])
                if verbose:
                    print(f"{i}:\t{input_terms[i]}\t{source_list[i]}")
                insignificant_terms.append(
                    (selected_input_indices[i], abs(ave_coeffs[i]))
                )  # diffs.append(abs(ave_coeffs[i]) - abs(error_bars[i]))

        # for removing smallest significant terms
        if insignificant_terms == [] and remove_significant:
            for i in range(len(ave_coeffs)):
                # print("{:.2}- {:.2}", ave_coeffs[i], error_bars[i])
                print(f"{i}:\t{input_terms[i]}\t{source_list[i]}")
                insignificant_terms.append(
                    (selected_input_indices[i], abs(ave_coeffs[i]))
                )  # diffs.append(abs(ave_coeffs[i]) - abs(error_bars[i]))

        # Now sort in order of ave_coeff
        insignificant_terms = sorted(insignificant_terms, key=lambda x: x[1])
        print(f"Insignificant Terms: {insignificant_terms}")

        # Now find the first term we can remove (if any)
        have_removed = False

        for idx, error_value in insignificant_terms:
            # If it has dependents, and you're doing an heirarchical model skip it
            if do_hierarchical:
                if dependency_dict[idx]:
                    continue
            print(
                f"removing term {input_terms[idx]} ({idx}) with error {error_value}"
            )
            output_indices.remove(idx)
            output_terms.remove(output_terms[idx])
            have_removed = True
            break

        print(f"output_indices are {output_indices}")
        terms.append(output_indices)
    return (
        output_indices,
        this_model,
    )


def map_chemical_space(
    unscaled_model,
    x_key,
    y_key,
    c_key,
    x_limits,
    y_limits,
    constant,
    n_points,
    hook_function,
):
    min_x = x_limits[0]
    max_x = x_limits[1]
    min_y = y_limits[0]
    max_y = y_limits[1]

    x = np.linspace(min_x, max_x, n_points)
    y = np.linspace(min_y, max_y, n_points)
    # c = np.linspace(constant, constant, n_points)

    mesh_x, mesh_y = np.meshgrid(x, y)

    z_df = pd.DataFrame()
    z_df[x_key] = mesh_x.reshape(-1)
    z_df[y_key] = mesh_y.reshape(-1)
    z_df[c_key] = constant
    # add any extra terms
    z_df = hook_function(z_df)

    mesh_z = unscaled_model.predict(z_df).reshape(n_points, n_points)

    return mesh_x, mesh_y, mesh_z


def map_chemical_space_new(
    unscaled_model,
    x_key,
    y_key,
    c_key,
    x_limits,
    y_limits,
    constant,
    n_points,
    hook_function,
    model,
    inputs,
    input_selector,
    source_list=None,
):
    if source_list is None:
        source_list = []
    min_x = x_limits[0]
    max_x = x_limits[1]
    min_y = y_limits[0]
    max_y = y_limits[1]

    x = np.linspace(min_x, max_x, n_points)
    y = np.linspace(min_y, max_y, n_points)
    # c = np.linspace(constant, constant, n_points)

    mesh_x, mesh_y = np.meshgrid(x, y)
    z_df = pd.DataFrame()
    z_df[x_key] = mesh_x.reshape(-1)
    z_df[y_key] = mesh_y.reshape(-1)
    z_df[c_key] = constant
    z_df = hook_function(z_df)

    mesh_z = unscaled_model.predict(z_df).reshape(n_points, n_points)

    return mesh_x, mesh_y, mesh_z


def add_higher_order_terms(
    inputs: pd.DataFrame,
    add_squares: bool = True,
    add_interactions: bool = True,
    column_list: list = [],
    verbose: bool = True,
):
    """Adds in squares and interactions terms
    inputs: the input/feature/variable array with data
    add_squares=True : whether to add square terms, e.g. x_1^2, x_2^2
    add_interactions=True: whether to add interaction terms, x_1*x_2, etc
    column_list=[]: to select only a subset of columns, input a column list here
    Currently does not go above power of 2

    returns saturated array and a list of which inputs created which column"""

    sat_inputs = copy.deepcopy(inputs)
    if not column_list:
        # do all columns
        column_list = [x for x in inputs.columns]
    if verbose:
        print(f"Input array has columns {column_list}")

    source_list = [x for x in column_list]

    if add_squares:
        if verbose:
            print("Adding square terms:")
        for i in range(len(column_list)):
            source_list.append(i)
            input_name = column_list[i]
            new_name = input_name + "**2"
            if verbose:
                print(new_name)
            sat_inputs[new_name] = inputs[input_name] * inputs[input_name]

    if add_interactions:
        if verbose:
            print("Adding interaction terms:")
        for i in range(len(column_list)):
            for j in range(i + 1, len(column_list)):
                source_list.append([i, j])
                input_name_1 = column_list[i]
                input_name_2 = column_list[j]
                new_name = input_name_1 + "*" + input_name_2
                if verbose:
                    print(new_name)
                sat_inputs[new_name] = (
                    inputs[input_name_1] * inputs[input_name_2]
                )

    return sat_inputs, source_list


def predict_from_model(model, inputs, input_selector):
    """Reorgs the inputs and does a prediction
    model = the model to use
    inputs = the saturated inputs
    input_selector = the subset of inputs the model is using
    """
    list_of_terms = [inputs.columns[x] for x in input_selector]
    model_inputs = inputs[list_of_terms]
    predictions = model.predict(model_inputs)
    return predictions, list_of_terms
