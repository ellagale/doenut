"""
DOENUT
Design of Experiments Numerical Utility Toolkit
"""

# first we import some useful libraries
import logging
import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
from typing import Tuple, List, TYPE_CHECKING, Any
from doenut.utils import initialise_log
from doenut.data import ModifiableDataSet
from doenut.models import AveragedModel


if TYPE_CHECKING:
    import sklearn


logger = initialise_log(__name__, logging.DEBUG)


def set_log_level(level: "str|int") -> None:
    """Sets the global log level for the module

    Parameters
    ----------
    level : "str|int"
        logging module value representing the desired log level
    """
    loggers = [
        logger
        for name, logger in logging.root.manager.loggerDict.items()
        if name.startswith("doenut")
    ]

    for l in loggers:
        if isinstance(l, logging.PlaceHolder):
            # can't set placeholders
            continue
        l.setLevel(level)


def orthogonal_scaling(
    inputs: pd.DataFrame, axis: int = 0
) -> Tuple[pd.DataFrame, float, float]:
    """Calculates the orthoganal scaling of an array along an axis

    Parameters
    ----------
    inputs : pd.DataFrame
        the dataframe to scale
    axis : int, default 0
        the axis to scale around (defaults to 0)

    Returns
    -------
    pd.DataFrame:
        The scaled inputs
    float:
        the Mj scaling parameter
    float:
        the Rj scaling parameter
    """
    # the scaling thingy that Modde uses
    inputs_max = np.max(inputs, axis)
    inputs_min = np.min(inputs, axis)
    Mj = (inputs_min + inputs_max) / 2
    Rj = (inputs_max - inputs_min) / 2
    scaled_inputs = (inputs - Mj) / Rj
    return scaled_inputs, Mj, Rj


def scale_1D_data(scaler, data, do_fit=True):
    """ELLA TODO: What does this do what it does?

    Parameters
    ----------
    scaler :
        the scaler to transform the data with
    data :
        the data to scale
    do_fit :
        whether to fit the data first (default true)

    Returns
    -------
    pd.DataFrame:
        The scaled data
    sklearn.scalar?
        The scaler object
    """
    if do_fit:
        scaler.fit(data.reshape(-1, 1))
    data_scaled = scaler.transform(data.reshape(-1, 1))
    data_scaled = data_scaled.reshape(-1)
    return data_scaled, scaler


def scale_by(new_data: pd.DataFrame, mj: float, rj: float) -> pd.DataFrame:
    """Scales a dataframe orthogonally using the supplied parameters according to
    the equation::

        result = (data - Mj) / Rj

    Parameters
    ----------
    new_data : pd.DataFrame
        the data to scale
    mj : float
        the Mj parameter
    rj : float
        the Rj parameter

    Returns
    -------
    pd.DataFrame
        the scaled data
    """
    # the scaling thingy that Modde uses
    # TODO:: Any form of sanity checking whatsoever.
    new_data = (new_data - mj) / rj
    return new_data


def find_replicates(inputs: pd.DataFrame) -> np.array:
    """Find experimental settings that are replicates

    Parameters
    ----------
    inputs : pd.DataFrame
        The dataframe to parse

    Returns
    -------
    np.array:
        A series of indices of all the rows which are replicates

    """
    # list comps ftw!
    a = [x for x in inputs[inputs.duplicated()].index]
    b = [x for x in inputs[inputs.duplicated(keep="last")].index]
    replicate_row_list = np.unique(np.array(a + b))
    return replicate_row_list


def train_model(
    inputs: pd.DataFrame,
    responses: pd.DataFrame,
    test_responses: pd.DataFrame,
    do_scaling_here: bool = False,
    fit_intercept: bool = False,
) -> Tuple["sklearn.linear_model", pd.DataFrame, float, List[Any]]:
    """A simple function to train a model

    Parameters
    ----------
    inputs :
        full set of terms for the model (x_n)
    responses :
        expected responses for the inputs (ground truth, y)
    test_responses :
        expected responses for separate test data (if used)
    do_scaling_here :
        whether to scale the data (Default value = False)
    fit_intercept :
        whether to fit the intercept (Default value = False)

    Returns
    -------
    sklearn.linear_model:
        A model fitted to the data,
    pd.DataFrame:
        the inputs used
    float:
        the R2 of that model
    List[Any]:
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
    logger.debug("R squared for this model is {:.3}".format(R2))
    return model, inputs, R2, predictions


def Calculate_R2(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    key: str,
    word: str = "test",
) -> float:
    """Calculates R2 from input data
    You can use this to calculate q2 if you're
    using the test ground truth as the mean
    else use calculate Q2
    I think this is what Modde uses for PLS fitting

    Parameters
    ----------
    ground_truth :
        The actual response values
    predictions :
        What the model guessed as the response values
    key :
        the column name into ground_truth that we predicted
    word :
        What mode we were working on
    ground_truth: pd.DataFrame :

    predictions: pd.DataFrame :

    key: str :

    word: str :
         (Default value = "test")

    Returns
    -------
    type
        the R2 of the model on this data, or the Q2 if in test mode.

    """
    errors = ground_truth[[key]] - predictions[[key]]
    test_mean = np.mean(ground_truth[[key]], axis=0)
    logger.debug(f"Mean of {word} set: {test_mean[0]}")
    errors["squared"] = errors[key] * errors[key]
    sum_squares_residuals = sum(errors["squared"])
    logger.debug(
        "Sum of squares of the residuals (explained variance) is"
        f"{sum_squares_residuals}"
    )
    sum_squares_total = sum((ground_truth[key] - test_mean[0]) ** 2)
    logger.debug(
        f"Sum of squares total (total variance) is {sum_squares_total}"
    )
    r2 = 1 - (sum_squares_residuals / sum_squares_total)
    if word == "test":
        logger.debug("{} is {:.3}".format("Q2", r2))
    else:
        logger.debug("{} is {:.3}".format("R2", r2))
    return r2


def Calculate_Q2(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    train_responses: pd.DataFrame,
    key: str,
    word: str = "test",
) -> float:
    """A different way of calculating Q2
    this uses the mean from the training data, not the
    test ground truth

    Parameters
    ----------
    ground_truth :
        The actual response values of the test set
    predictions :
        The predictions of the model for the test set
    train_responses :
        The response values of the training set
    key :
        Which column in the ground_truth we are predicting
    word :
        The mode to run in
    ground_truth: pd.DataFrame :

    predictions: pd.DataFrame :

    train_responses: pd.DataFrame :

    key: str :

    word: str :
         (Default value = "test")

    Returns
    -------
    type
        The calculated Coefficient (R2/Q1)

    """
    errors = ground_truth[[key]] - predictions[[key]]
    train_mean = np.mean(train_responses[[key]], axis=0)
    test_mean = np.mean(ground_truth[[key]], axis=0)
    logger.debug(f"Mean of {word} set: {test_mean.iloc[0]}")
    logger.debug(f"Mean being used: {train_mean.iloc[0]}")
    errors["squared"] = errors[key] * errors[key]
    sum_squares_residuals = sum(errors["squared"])
    logger.debug(
        "Sum of squares of the residuals (explained variance) is"
        f"{sum_squares_residuals}"
    )
    sum_squares_total = sum((ground_truth[key] - train_mean.iloc[0]) ** 2)
    # stuff from Modde
    # errors/1

    logger.debug(
        f"Sum of squares total (total variance) is {sum_squares_total}"
    )
    result = 1 - (sum_squares_residuals / sum_squares_total)
    logger.info(f"{'Q2' if word=='test' else 'R2'} is {round(result,3)}")
    return result


def dunk(setting: "str|None" = None) -> None:
    """dunk your doenut

    Parameters
    ----------
    setting : str, default None
        what you are dunking it into

    """
    if setting == "coffee":
        print("Splash!")
    else:
        print("Requires coffee")
    return


def average_replicates(
    inputs: pd.DataFrame, responses: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """averages inputs that are the same

    Parameters
    ----------
    inputs :
        The input data to average
    responses :
        The responses to averaged
    inputs: pd.DataFrame :

    responses: pd.DataFrame :


    Returns
    -------
    type
        A tuple of the averaged inputs and responses

    """
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
                logger.debug(
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
    approximated by 2*std)

    Parameters
    ----------
    coeffs :
        The coefficents to calculate from
    labels :
        No longer used?
    errors :
        The type of error to calculate, C{std} or C{p95} (Default value = "std")
    normalise :
        Whether to normalise the data prior to calculation (Default value = False)

    Returns
    -------
    type
        A tuple of the averaged coefficients and their error bars

    """

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
):
    """Attempts to automatically tune a parsimonious model

    TODO:: update to new code and remove redundant parameters

    Parameters
    ----------
    inputs :
        The input data to train on
    responses :
        The response values for the input data
    source_list :
        param response_selector: (Optional) Which columns in responses to use
    use_scaled_inputs :
        Optional) Whether to scale the inputs before calculations (Default value = True)
    do_scaling_here :
        Optional) Whether to scale each set of train/test data (Default value = True)
    drop_duplicates :
        Optional) Do we ingnore (C{'no'}), C{'average'}, C{'Drop'} duplicate input values (Default value = "average")
    errors :
        Optional) C{'p95'} for 95th percentile or C{'std'} for standard deviation for error calculation (Default value = "p95")
    normalise :
        Optional) Whether to normalise the coefficents for error calculation (Default value = True)
    do_hierarchical :
        Optional) Do we maintain a hierarchical model? (Default value = True)
    remove_significant :
        Optional) Model will continue removing terms until only one is left (Default value = False)
    response_selector :
         (Default value = [0])

    Returns
    -------
    type
        A tuple of the terms used in the final model and the final model.

    """
    sat_inputs = inputs

    logger.debug(f"Source list is {source_list}")
    input_selector = [i for i in range(len(sat_inputs.columns))]
    output_indices = input_selector
    # global list of all column names.
    input_terms = list(sat_inputs.columns)
    output_terms = input_terms
    logger.debug("numbers\tnames")
    for i, v in enumerate(input_terms):
        logger.debug(f"{i}\t{v}")
    this_model = None
    have_removed = True
    R2_over_opt = []
    Q2_over_opt = []
    n_terms_over_opt = []
    terms = []

    while have_removed:
        logger.info("Beginning loop")
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
        logger.info(f"Selected terms {selected_input_terms}")
        logger.info(f"Source List: {source_list}")
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
                        except KeyError as e:
                            if do_hierarchical:
                                logger.error(
                                    "Error: Hierarchical model missing lower level terms."
                                )
                            raise e
        logger.info(f"Dependencies: {dependency_dict}")
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
                logger.debug(f"{i}:\t{input_terms[i]}\t{source_list[i]}")
                insignificant_terms.append(
                    (selected_input_indices[i], abs(ave_coeffs[i]))
                )  # diffs.append(abs(ave_coeffs[i]) - abs(error_bars[i]))

        # for removing smallest significant terms
        if insignificant_terms == [] and remove_significant:
            for i in range(len(ave_coeffs)):
                # print("{:.2}- {:.2}", ave_coeffs[i], error_bars[i])
                logger.info(f"{i}:\t{input_terms[i]}\t{source_list[i]}")
                insignificant_terms.append(
                    (selected_input_indices[i], abs(ave_coeffs[i]))
                )  # diffs.append(abs(ave_coeffs[i]) - abs(error_bars[i]))

        # Now sort in order of ave_coeff
        insignificant_terms = sorted(insignificant_terms, key=lambda x: x[1])
        logger.info(f"Insignificant Terms: {insignificant_terms}")

        # Now find the first term we can remove (if any)
        have_removed = False

        for idx, error_value in insignificant_terms:
            # If it has dependents, and you're doing a hierarchical model skip it
            if do_hierarchical:
                if dependency_dict[idx]:
                    continue
            logger.info(
                f"removing term {input_terms[idx]} ({idx}) with error {error_value}"
            )
            output_indices.remove(idx)
            output_terms.remove(output_terms[idx])
            have_removed = True
            break

        logger.info(f"output_indices are {output_indices}")
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
    """Calculates a three way map of chemical space for plotting

    #TODO:: Should move this to doenut.plot

    Parameters
    ----------
    unscaled_model :
        The model to plot
    x_key :
        What key to use for the X axis
    y_key :
        What key to use for the Y axis
    c_key :
        What key to use for the C axis
    x_limits :
        Tuple of min/max range of X to plot
    y_limits :
        Tuple of min/max range of y to plot
    constant :
        The value for C
    n_points :
        How many marks along each axis to generate
    hook_function :
        A custom data processing function for post processing the data

    Returns
    -------
    type
        Three meshes of the model's predictions for the keys/ranges predicted.

    """
    min_x = x_limits[0]
    max_x = x_limits[1]
    min_y = y_limits[0]
    max_y = y_limits[1]

    x = np.linspace(min_x, max_x, n_points)
    y = np.linspace(min_y, max_y, n_points)

    mesh_x, mesh_y = np.meshgrid(x, y)

    z_df = pd.DataFrame()
    z_df[x_key] = mesh_x.reshape(-1)
    z_df[y_key] = mesh_y.reshape(-1)
    z_df[c_key] = constant
    # add any extra terms
    z_df = hook_function(z_df)

    mesh_z = unscaled_model.predict(z_df).reshape(n_points, n_points)

    return mesh_x, mesh_y, mesh_z


def add_higher_order_terms(
    inputs: pd.DataFrame,
    add_squares: bool = True,
    add_interactions: bool = True,
    column_list: list = [],
) -> Tuple[pd.DataFrame, List]:
    """Generate a saturated set of inputs by adding the power and interaction terms
    Currently does not go above power of 2

    Parameters
    ----------
    inputs :
        The data to generate from
    add_squares :
        Optional) Whether to add square terms, e.g. x_1*2
    add_interactions :
        Optional) Whether to add interaction terms, e.g. x_1*x_2
    column_list :
        Optional) Which columns to generate from
    inputs: pd.DataFrame :

    add_squares: bool :
         (Default value = True)
    add_interactions: bool :
         (Default value = True)
    column_list: list :
         (Default value = [])

    Returns
    -------
    type
        Tuple of the saturated inputs, and a list of which inputs created
        which input column.

    """

    sat_inputs = copy.deepcopy(inputs)
    if not column_list:
        # do all columns
        column_list = [x for x in inputs.columns]
    logger.debug(f"Input array has columns {column_list}")

    source_list = [x for x in column_list]

    if add_squares:
        logger.debug("Adding square terms:")
        for i in range(len(column_list)):
            source_list.append(i)
            input_name = column_list[i]
            new_name = input_name + "**2"
            logger.debug(new_name)
            sat_inputs[new_name] = inputs[input_name] * inputs[input_name]

    if add_interactions:
        logger.debug("Adding interaction terms:")
        for i in range(len(column_list)):
            for j in range(i + 1, len(column_list)):
                source_list.append([i, j])
                input_name_1 = column_list[i]
                input_name_2 = column_list[j]
                new_name = input_name_1 + "*" + input_name_2
                logger.debug(new_name)
                sat_inputs[new_name] = (
                    inputs[input_name_1] * inputs[input_name_2]
                )

    return sat_inputs, source_list


def predict_from_model(model, inputs, input_selector):
    """Reorgs the inputs and does a prediction

    Parameters
    ----------
    model :
        the model to use
    inputs :
        the saturated inputs
    input_selector :
        the subset of inputs the model is using

    Returns
    -------
    type
        Tuple of the predictions and the terms used to generate them

    """
    list_of_terms = [inputs.columns[x] for x in input_selector]
    model_inputs = inputs[list_of_terms]
    predictions = model.predict(model_inputs)
    return predictions, list_of_terms
