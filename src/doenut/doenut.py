"""
DOENUT
Design of Experiments Numerical Utility Toolkit
"""

# first we import some useful libraries
import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression

import doenut
from doenut.models.averaged_model import AveragedModel


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


def average_replicates(inputs, responses, verbose=False):
    """averages inputs that are the same
    TO-DO - you can make it pick nearly teh same inputs if
    if you add the actual values which are not always the expected values
    inputs = inputs
    responses = responses"""

    whole_inputs = inputs
    duplicates = [x for x in whole_inputs[whole_inputs.duplicated()].index]
    duplicates_for_averaging = {}
    non_duplicate_list = [x for x in whole_inputs.index if x not in duplicates]
    for non_duplicate in non_duplicate_list:
        this_duplicate_list = []
        non_duplicate_row = whole_inputs.loc[[non_duplicate]].to_numpy()
        for duplicate in duplicates:
            duplicate_row = whole_inputs.loc[[duplicate]].to_numpy()
            if (non_duplicate_row == duplicate_row).all():
                this_duplicate_list.append(duplicate)
                if verbose:
                    print(
                        f"found duplicate pairs: {non_duplicate}, {duplicate}"
                    )
        duplicates_for_averaging[non_duplicate] = this_duplicate_list

    averaged_responses = []
    averaged_inputs = []

    for non_duplicate, duplicates in duplicates_for_averaging.items():
        # print(f"nd: {non_duplicate}")
        to_average = whole_inputs.loc[[non_duplicate]]
        to_average_responses = responses.loc[[non_duplicate]]
        for duplicate in duplicates:
            to_average = to_average.append(whole_inputs.loc[[duplicate]])
            to_average_responses = to_average_responses.append(
                responses.loc[[duplicate]]
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


def calc_averaged_model(
    inputs,
    responses,
    key="",
    drop_duplicates="Yes",
    fit_intercept=False,
    do_scaling_here=False,
    use_scaled_inputs=False,
):
    """Use 'leave one out' method to train and test a series of models.

    :param inputs: full set of terms for the model (x_n)
    :param responses: responses to model (ground truth, y)
    :param key: Method used to calculate Q2, e.g. 'ortho'
    :param drop_duplicates: what to do with duplicates:
        'Yes' or True: drop them
        'average': average their values
        otherwise: leave them in
    :param fit_intercept: whether to fit the intercept
    :param do_scaling_here: whether to scale the data
    :param use_scaled_inputs:
    :return: A tuple of the following:
        The ortho (averaged) model.
        Predictions from the model for the inputs.
        The inputs measured (the ground truth).
        The coefficients of the models.
        A list of the R2 values for the models.
        The R2 correlation coefficient for the ortho model.
        The Q2 correlation coefficient for the ortho model.
    """

    # first we copy the data sideways
    if use_scaled_inputs:
        inputs, _, _ = orthogonal_scaling(inputs, axis=0)
    whole_inputs = inputs
    whole_responses = responses
    if (drop_duplicates == "Yes") or (
        type(drop_duplicates) == bool and drop_duplicates
    ):
        print("Dropping replicates")
        duplicates = [x for x in whole_inputs[whole_inputs.duplicated()].index]
    elif drop_duplicates == "average":
        # averages the replicates
        print("Averaging replicates")
        whole_inputs, whole_responses = average_replicates(inputs, responses)
        duplicates = []
    else:
        print("Have found no replicates")
        duplicates = []
    # we drop the duplicates
    inputs = whole_inputs.drop(duplicates)
    responses = whole_responses.drop(duplicates)

    predictions = []
    ground_truth = []
    errors = []
    coeffs = []
    intercepts = []
    R2s = []
    print(f"Input data is {len(whole_inputs)} points long")
    print(f"We are using {len(inputs)} data points")
    for idx, i in enumerate([x for x in inputs.index]):
        # print(i)
        # pick a row of the dataset and make that the test set
        test_input = inputs.iloc[idx].to_numpy().reshape(1, -1)
        test_response = responses.iloc[idx]
        # drop that row from the training data
        train_inputs = inputs.drop(i).to_numpy()
        train_responses = responses.drop(i)
        # here we instantiate the model
        model, _, _, _ = train_model(
            train_inputs,
            train_responses,
            test_responses=None,
            fit_intercept=fit_intercept,
            do_scaling_here=do_scaling_here,
            verbose=False,
        )
        R2 = model.score(train_inputs, train_responses)
        # we save the coordinates
        coeffs.append(model.coef_)
        intercepts.append(model.intercept_)
        # use the model to predict on the test set and get errors
        prediction = model.predict(test_input)
        prediction = prediction[0]
        error = test_response - prediction
        # error = error[0][0]
        print(
            "Left out data point {}:\tR2 = {:.3}\tAve. Error = {:.3}".format(
                i, R2, np.mean(error)
            )
        )
        # this saves the data for this model
        predictions.append(prediction)
        ground_truth.append(test_response)
        errors.append(error)
        # predictions_out = np.array(predictions)
        # test_responses = np.array(ground_truth)

        R2s.append(R2)
    # these coefficients is what defines the final model
    egg = np.array(coeffs)
    averaged_coeffs = np.array(np.mean(egg, axis=0))
    # here we overwrite the final model with the averaged coefficients
    model.coef_ = averaged_coeffs
    # same with intercepts
    average_intercept = np.mean(intercepts, axis=0)
    model.intercept_ = average_intercept
    # and score the model, this is for all responses together
    R2 = model.score(whole_inputs.to_numpy(), whole_responses.to_numpy())
    print("R2 overall is {:.3}".format(R2))
    df_pred = pd.DataFrame.from_records(predictions, columns=responses.columns)
    df_GT = pd.DataFrame.from_records(ground_truth, columns=responses.columns)
    Q2 = Calculate_Q2(
        ground_truth=df_GT,
        predictions=df_pred,
        train_responses=whole_responses,  # is it this one?
        word="test",
        key=key,
        verbose=True,
    )
    doenut.plot.plot_summary_of_fit_small(R2, Q2)

    return model, df_pred, df_GT, coeffs, R2s, R2, Q2


def calc_ave_coeffs_and_errors(coeffs, labels, errors="std", normalise=False):
    """Coefficient plot
    set error to 'std' for standard deviation
    set error to 'p95' for 95th percentile (
    approximated by 2*std)"""

    ave_coeffs = np.mean(coeffs, axis=0)[0]
    stds = np.std(coeffs, axis=0)[0]
    if normalise:
        ave_coeffs = ave_coeffs / stds
        stds = stds / stds  # stds = np.std(coeffs, axis=0)[0]
    if errors == "std":
        error_bars = stds
    elif errors == "p95":
        # this is an approximation assuming a gaussian distribution in your coeffs
        error_bars = 2 * stds
    else:
        print(f"Error: errors setting {errors} not known, chose std or p95")

    return ave_coeffs, error_bars


def calculate_R2_and_Q2_for_models(
    inputs,
    responses,
    input_selector=None,
    response_selector=None,
    fit_intercept=True,
    use_scaled_inputs=False,
    do_scaling_here=False,
    drop_duplicates="average",
    do_plot=True,
    do_r2=True,
    verbose=True,
):
    """Calculates R2 for model, sub-models
    and allows removal of terms
    Can be called to loop over all responses
    Or can be called on one response to allow for dropping terms
    inputs: inputs dataframe
    responses: response dataframe
    model: trained model
    input_selector: list of term numbers (column numbers) in input you want
    response_selector: list of column numbers in response dataframe you want
    use_scaled_inputs: this adds a bias term to the model
    do_scaling_here: whether to calculate the scaling inside this function or not
    """
    # if use_scaled_inputs:
    #    inputs = orthogonal_scaling(inputs)
    # finds out which columns we're going to use
    input_column_list = list(inputs.columns)
    response_column_list = list(responses.columns)
    if verbose:
        print(f"Input terms are {input_column_list}")
        print(f"Input Responses are {response_column_list}\n")
    # make a linear regression model
    if response_selector is None:
        # do all columns
        res_col_num_list = [x for x in range(len(responses.columns))]
    else:
        res_col_num_list = range(len(response_selector))
    if input_selector is None:
        # saturated model - do all columns
        input_selector = range(len(input_column_list))
    for res_col_num in response_selector:
        # loops over responses, to get R2 for all responses at once
        # don't use this function
        # finds key
        response_key = response_column_list[res_col_num]
        if verbose:
            print(f"Selected Response is {response_key}")
        # creates a new response dataframe of just the response requested
        this_model_responses = responses[[response_key]]
        # creates a new input dataframe of just the inputs desired
        edited_input_data = inputs.iloc[:, input_selector]
        selected_input_terms = [x for x in edited_input_data.columns]
        print("Selected input terms:\t{}".format(selected_input_terms))
        # makes a new model
        # temp_tuple = calc_averaged_model(
        #     edited_input_data,
        #     this_model_responses,
        #     key=response_key,
        #     drop_duplicates=drop_duplicates,
        #     fit_intercept=fit_intercept,
        #     use_scaled_inputs=use_scaled_inputs,
        #     do_scaling_here=do_scaling_here,
        # )

        model = AveragedModel(
            edited_input_data,
            this_model_responses,
            use_scaled_inputs,
            fit_intercept,
            response_key,
            drop_duplicates,
        )

        temp_tuple = (
            model.model,
            model.df_pred,
            model.df_gt,
            model.coeffs,
            model.r2s,
            model.get_r2(),
            model.q2,
        )

        new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
        # we fit it as this is hte easiest way to set up the new model correctly
        new_model.fit(edited_input_data, this_model_responses)
        # we overwrite the coefficients with the averaged model
        if len(res_col_num_list) == 1:
            coefficient_list = new_model.coef_[0]
        else:
            coefficient_list = new_model.coef_[res_col_num]
        # selected_coefficient_list = coefficient_list[input_selector]
        # print("Coefficients:\t{}".format(selected_coefficient_list))
        new_model.coef_ = coefficient_list
        # now get the R2 value
        R2 = new_model.score(edited_input_data, this_model_responses)
        if verbose:
            print("Response {} R2 is {:.3}".format(response_key, R2))
            print("Input selector was {}".format(input_selector))
        if not do_r2:
            doenut.plot.clear_figure()
        if do_plot:
            doenut.plot.coeff_plot(
                coeffs,
                labels=selected_input_terms,
                errors="p95",
                normalise=True,
            )
            # print(averaged_coeffs)
            # scaled_coeffs = orthogonal_scaling(coefficient_list)
            # plt.bar([x for x in range(len(scaled_coeffs))],scaled_coeffs)

    return new_model, R2, temp_tuple, selected_input_terms


def tune_model(
    inputs,
    responses,
    input_selector=None,
    response_selector=None,
    fit_intercept=True,
    use_scaled_inputs=False,
    do_scaling_here=False,
    drop_duplicates="average",
    do_plot=True,
    do_r2=True,
    verbose=True,
):
    """Wrapper to calculate_R2_and_Q2_for_models to make life easy
    It does both scaled and unscaled models
    assumes you want an unscaled model for ease of plotting
    and a scaled model coefficients for ease of choosing"""

    # scaled model, use this for picking your coefficients
    (
        this_model,
        R2,
        temp_tuple,
        selected_input_terms,
    ) = calculate_R2_and_Q2_for_models(
        inputs,
        responses,
        input_selector=input_selector,
        response_selector=response_selector,
        use_scaled_inputs=True,
        drop_duplicates="No",
        do_scaling_here=True,
    )
    scaled_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple

    # unscaled model, use this for picking your coefficients
    # this_model, R2, temp_tuple, selected_input_terms = calculate_R2_and_Q2_for_models(
    #                    inputs,
    #                    responses[['Profit']],
    #                    input_selector=input_selector,
    #                    response_selector=response_selector,
    #                    use_scaled_inputs=False,
    #                    drop_duplicates='No',
    #                    do_scaling_here=False,
    #                    do_plot=False,
    #                    verbose=False)
    # unscaled_model, predictions, ground_truth, coeffs, R2s, R2, Q2= temp_tuple
    # unscaled_model = []

    return scaled_model, R2, temp_tuple, selected_input_terms


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
    if verbose:
        print("numbers\tnames")
        for i, v in enumerate(input_terms):
            print(f"{i}\t{v}")
    have_removed = True
    R2_over_opt = []
    Q2_over_opt = []
    n_terms_over_opt = []
    terms = []

    while have_removed:
        selected_input_indices = output_indices

        # print(selected_input_indices)
        # print(sat_inputs)
        if len(selected_input_indices) == 0:
            break
        (
            this_model,
            R2,
            temp_tuple,
            selected_input_terms,
        ) = calculate_R2_and_Q2_for_models(
            sat_inputs,
            responses,
            input_selector=selected_input_indices,
            response_selector=[0],
            use_scaled_inputs=True,
            do_scaling_here=True,
            drop_duplicates=drop_duplicates,
            do_r2=False,
            verbose=False,
        )
        new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple

        R2_over_opt.append(R2)
        Q2_over_opt.append(Q2)

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
            coeffs=coeffs,
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
        print(insignificant_terms)

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
            have_removed = True
            break

        print(f"output_indices are {output_indices}")
        terms.append(output_indices)
    return (
        output_indices,
        new_model,
        predictions,
        ground_truth,
        coeffs,
        R2s,
        R2,
        Q2,
        R2_over_opt,
        Q2_over_opt,
        n_terms_over_opt,
        terms,
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

    # sat_inputs = add_higher_order_terms(
    #         z_df,
    #         add_squares=True,
    #         add_interactions=True,
    #         column_list=source_list,
    #         verbose=True,
    #     )
    # predict_from_model(model, inputs, input_selector)

    mesh_z = unscaled_model.predict(z_df).reshape(n_points, n_points)

    return mesh_x, mesh_y, mesh_z


def add_higher_order_terms(
    inputs,
    add_squares=True,
    add_interactions=True,
    column_list=[],
    verbose=True,
):
    """Adds in squares and interactions terms
    inputs: the input/feature/variable array with data
    add_squares=True : whether to add square terms, e.g. x_1^2, x_2^2
    add_interactions=True: whether to add interaction terms, x_1*x_2, etc
    column_list=[]: to select only a subset of columns, input a column list here
    Currently does not go above power of 2

    returns saturated array and a list of which inputs created which column"""

    sat_inputs = copy.deepcopy(inputs)
    if column_list == []:
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
