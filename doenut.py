#############################################################
#															#
#							DOENUT							#
#	Design Of Experiments ...
#															#
#############################################################

# first we import some useful libraries
import numpy as np
import pandas as pd
import copy
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

def orthogonal_scaling(inputs):
    ## the scaling thingy that Modde uses
    inputs_max = np.max(inputs)
    inputs_min = np.min(inputs)
    Mj = (inputs_min + inputs_max)/2
    Rj = (inputs_max - inputs_min)/2
    scaled_inputs = (inputs - Mj)/Rj
    return scaled_inputs, Mj, Rj

def scale_1D_data(scaler, data, do_fit=True):
    if do_fit:
        scaler.fit(data.reshape(-1,1))
    data_scaled = scaler.transform(data.reshape(-1,1))
    data_scaled = data_scaled.reshape(-1)
    return data_scaled, scaler

def applying_orthogonal_scaling_to_new_data(new_data):
    ## the scaling thingy that Modde uses
    new_data = (new_data - Mj)/Rj
    return new_data

def find_replicates(inputs):
    """Find experimental settings that are replicates"""
    # list comps ftw!
    a=[x for x in inputs[inputs.duplicated()].index]
    b=[x for x in inputs[inputs.duplicated(keep='last')].index]
    replicate_row_list = np.unique(np.array(a+b))
    return replicate_row_list

def replicate_plot(inputs, responses, key):
    """Plots a replicate plot which shows all experiments
    and identifies the replicates
    inputs: 
    responses:
    key: column in responses that you wish to plot"""
    plt.title(key)
    replicate_row_list = find_replicates(inputs)
    non_replicate_row_list = [x for x in range(len(responses)) if x not in replicate_row_list]
    col = responses[key]
    x_data = [x for x in range(len(non_replicate_row_list))]
    y_data = [col.loc[x] for x in non_replicate_row_list]
    plt.plot(x_data,
             y_data,'o')
    ax = plt.gca()
    ax.set_xticks(x_data)
    ax.set_xticklabels(non_replicate_row_list)
    plt.xlabel('Experiment No.')
    plt.ylabel('Response')

    x_data = [len(non_replicate_row_list) for x in range(len(replicate_row_list))]
    y_data = [col.loc[x] for x in replicate_row_list]
    plt.plot(x_data, y_data,'or')
    return

def train_model(inputs, 
                responses, 
                test_responses,
                do_scaling_here=False,
                fit_intercept=False,
                verbose=True):
    """ Simple function to train models
    inputs: input matrix to the model
    responses:
    train_model returns:
    a model fitted to your data: original_model
    the inputs given to it: inputs_used
    the  𝑅2  of that model: original_model_R2
    a set of predictions that the model gives for your original data: predictions.
    The inputs to the function doenut.train_model() are:

    the inputs: inputs
    the outputs: responses['ortho']. By typing either 'ortho', 'para' or 'di' in the square brackets you can select the output you want to model.
    test_responses: if you have a separate set of test data you can input it here to try your model out on that
    do_scaling_here: whether to scale the data. You do not need to do this in this work.
    fit_intercept: whether or not to fit the intercept. In this work you always want to fit the intercept.
    verbose: this is a common setting in coding, if true it means run in verbose mode and many bits of information are printed to the screen."""
    if do_scaling_here:
        inputs, _, _ =orthogonal_scaling(inputs)
    if test_responses == None:
        test_responses = responses
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(inputs, responses)
    predictions = model.predict(inputs)
    R2 = model.score(inputs, test_responses)
    if verbose:
        print("R squared for this model is {:.3}".format(R2))
    return model, inputs, R2, predictions

def plot_observed_vs_predicted(responses, 
                               predictions,
                               range_x=[],
                               label='',
                               do_axes_equal = True):
    """plots a graph duh
    range should be in the form [min_x, max_x]
    else it will take from responses"""
    plt.plot(responses, predictions,'o')
    min_xy = np.min([np.min(responses),np.min(predictions)])
    max_xy = np.max([np.max(responses),np.max(predictions)])
    if range_x == []:
        range_x = [(min_xy//10)*10, (max_xy//10)*10+10]
    plt.plot(range_x,range_x)
    x_label = 'Measured ' + label
    y_label = 'Predicted ' + label
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if do_axes_equal:
        plt.gca().set_aspect('equal')
    return

def Calculate_R2(ground_truth, 
                 predictions, 
                 key,
                 word='test', 
                 verbose=True):
    """Calculates R2 from input data
    You can use this to calculate q2 if you're 
    using the test ground truth as the mean
    else use calculate Q2
    I think this is what Modde uses for PLS fitting"""
    errors = ground_truth[[key]] - predictions[[key]]
    test_mean = np.mean(ground_truth[[key]], axis=0)
    if verbose:
        print(f'Mean of {word} set: {test_mean[0]}')
    errors['squared'] = errors[key]*errors[key]
    sum_squares_residuals = sum(errors['squared'])
    if verbose:
        print(f'Sum of squares of the residuals (explained variance) is {sum_squares_residuals}')
    sum_squares_total = sum((ground_truth[key] - test_mean[0])**2)
    if verbose:
        print(f'Sum of squares total (total variance) is {sum_squares_total}')
    R2 = 1 - (sum_squares_residuals / sum_squares_total)
    if word == 'test':
        print("{} is {:.3}".format("Q2", R2))
    else:
        print("{} is {:.3}".format("R2", R2))
    return R2

def Calculate_Q2(ground_truth, 
                 predictions, 
                 train_responses,
                 key,
                 word='test', 
                 verbose=True):
    """A different way of calculating Q2
    this uses the mean from the training data, not the 
    test ground truth"""
    errors = ground_truth[[key]] - predictions[[key]]
    train_mean = np.mean(train_responses[[key]], axis=0)
    test_mean = np.mean(ground_truth[[key]], axis=0)
    if verbose:
        print(f'Mean of {word} set: {test_mean[0]}')
        print(f"Mean being used: {train_mean[0]}")
    errors['squared'] = errors[key]*errors[key]
    sum_squares_residuals = sum(errors['squared'])
    if verbose:
        print(f'Sum of squares of the residuals (explained variance) is {sum_squares_residuals}')
    sum_squares_total = sum((ground_truth[key] - train_mean[0])**2)
    ##### stuff from Modde
    #errors/1
    
    
    if verbose:
        print(f'Sum of squares total (total variance) is {sum_squares_total}')
    R2 = 1 - (sum_squares_residuals / sum_squares_total)
    if word == 'test':
        print("{} is {:.3}".format("Q2", R2))
    else:
        print("{} is {:.3}".format("R2", R2))
    return R2

def plot_summary_of_fit_small(R2, Q2):
    """Plots a nice graph of R2 and Q2"""
    my_colors = ['green','blue','pink', 'cyan',
                 'red', 'green','blue', 'cyan','orange','purple'] 
    plt.bar([1,2], [R2, Q2],color=my_colors)
    plt.grid(True)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
    #plt.ylabel('Average Price')
    plt.xticks([1,2],labels=["R2","Q2"])
    plt.ylim((0,1))
    plt.grid(axis = 'x')
    return

def dunk(setting=None):
    if setting == 'coffee':
        print('Splash!')
    else:
        print('Requires coffee')
    return

def average_replicates(
    inputs,
    responses,
    verbose=False):
    """averages inputs that are the same
    TO-DO - you can make it pick nearly teh same inputs if
    if you add the actual values which are not always the expected values
    inputs = inputs
    responses = responses"""


    whole_inputs =inputs
    duplicates = [x for x in whole_inputs[whole_inputs.duplicated()].index ]
    duplicates_for_averaging = {}
    non_duplicate_list = [x for x in whole_inputs.index if x not in duplicates]
    for non_duplicate in non_duplicate_list:
        this_duplicate_list=[]
        non_duplicate_row = np.array(whole_inputs.loc[[non_duplicate]])
        for duplicate in duplicates:
            duplicate_row = np.array(whole_inputs.loc[[duplicate]])
            if (non_duplicate_row==duplicate_row).all():
                this_duplicate_list.append(duplicate)
                if verbose:
                    print(f"found duplicate pairs: {non_duplicate}, {duplicate}")
        duplicates_for_averaging[non_duplicate]=this_duplicate_list

    averaged_responses = []
    averaged_inputs = []

    for non_duplicate, duplicates in duplicates_for_averaging.items():
        #print(f"nd: {non_duplicate}")
        to_average=whole_inputs.loc[[non_duplicate]]
        to_average_responses=responses.loc[[non_duplicate]]
        for duplicate in duplicates:
            to_average = to_average.append(whole_inputs.loc[[duplicate]])
            to_average_responses = to_average_responses.append(responses.loc[[duplicate]])
        meaned = to_average.mean(axis=0)
        meaned_responses = to_average_responses.mean(axis=0)
        try:
            averaged_inputs=averaged_inputs.append(pd.DataFrame(meaned).transpose(), ignore_index=True)
            averaged_responses=averaged_responses.append(pd.DataFrame(meaned_responses).transpose(), ignore_index=True)
        except TypeError:
            averaged_inputs=pd.DataFrame(meaned).transpose()
            averaged_responses=pd.DataFrame(meaned_responses).transpose()
    return averaged_inputs, averaged_responses   

def calc_averaged_model(inputs, 
                        responses, 
                        key='',
                        drop_duplicates='Yes',
                        fit_intercept=False,
                        do_scaling_here=False,
                        use_scaled_inputs=False):
    """Uses 'leave one out' method to train and test a series 
    of models
    inputs: full set of terms for the model (x_n)
    responses: responses to model (ground truth, y)
    drop-duplicates: True, we drop the duplicates on the leave one out model, 'average', we average them
    to avoid inflating the results
    doenut.calc_averaged_model usage details:

    Takes in:

    inputs: the input dataframe
    responses[['ortho']]: the response data for the ortho-product
    key='ortho': used to calculate Q2
    drop_duplicates: removes the replicate experiments
    others as above
    Outputs:

    the averaged model (called ortho_model below as it models the ortho-product data)
    predictions from the model from the inputs
    the inputs measured (called ground_truth below)
    coeffs: the coefficients of the models
    R2S: a list of the  𝑅2  values for each model
    R2: the  𝑅2  correlation coefficient on the training data for the averaged model (the fitting coefficient)
    Q2: the  𝑄2  correlation coefficient on the testing data for the averaged model (the predicting coefficient)"""
    # first we copy the data sideways
    if use_scaled_inputs:
        inputs, _, _ = orthogonal_scaling(inputs)
    whole_inputs = inputs
    whole_responses = responses
    if (drop_duplicates=='Yes') or (type(drop_duplicates)==bool and drop_duplicates):
        print('Dropping replicates')
        duplicates = [x for x in whole_inputs[whole_inputs.duplicated()].index ]
    elif drop_duplicates == 'average':
        # averages the replicates
        print('Averaging replicates')
        whole_inputs, whole_responses = average_replicates(
            inputs,
            responses)
        duplicates = []
    else:
        print('Have found no replicates')
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
    for idx,i in enumerate([x for x in inputs.index]):
        #print(i)
        # pick a row of the dataset and make that the test set
        test_input = np.array(inputs.iloc[idx]).reshape(1, -1)
        test_response = responses.iloc[idx]
        # drop that row from the training data
        train_inputs = inputs.drop(i)
        train_responses = responses.drop(i)
        # here we instantiate the model
        model, _, _,_ = train_model(train_inputs, 
                train_responses, 
                test_responses=None,
                fit_intercept=fit_intercept,
                do_scaling_here=do_scaling_here,
                verbose=False)
        R2 = model.score(train_inputs, train_responses)
        # we save the coordinates
        coeffs.append(model.coef_)
        intercepts.append(model.intercept_)
        # use the model to predict on the test set and get errors
        prediction = model.predict(test_input)
        prediction = prediction[0]
        error = test_response - prediction
        #error = error[0][0]
        print("Left out data point {}:\tR2 = {:.3}\tAve. Error = {:.3}".format(i, R2, np.mean(error)))
        # this saves the data for this model
        predictions.append(prediction)
        ground_truth.append(test_response)
        errors.append(error)
        predictions_out = np.array(predictions)
        test_responses = np.array(ground_truth)
        egg=np.array(coeffs)
        R2s.append(R2)
    # these coefficients is what defines the final model
    averaged_coeffs = np.array(np.mean(egg, axis=0))
    # here we overwrite the final model with the averaged coefficients
    model.coef_ = averaged_coeffs
    # same with intercepts
    average_intercept = np.mean(intercepts,axis=0)
    model.intercept_ = average_intercept
    # and score the model, this is for all responses together
    R2 = model.score(whole_inputs, whole_responses)
    print("R2 overall is {:.3}".format(R2))
    df_pred = pd.DataFrame.from_records(predictions, columns=responses.columns)
    df_GT = pd.DataFrame.from_records(ground_truth, columns=responses.columns)
    Q2 = Calculate_Q2(ground_truth=df_GT, 
                 predictions = df_pred, 
                 train_responses=whole_responses, ### is it this one?
                 word='test', 
                 key=key,
                 verbose=True)
    plot_summary_of_fit_small(R2, Q2)

    return model, df_pred, df_GT, coeffs, R2s, R2, Q2

def calc_ave_coeffs_and_errors(coeffs, labels, errors='std',normalise=False):
    """Coefficient plot
    set error to 'std' for standard deviation
    set error to 'p95' for 95th percentile (
    approximated by 2*std)"""
        
    ave_coeffs = np.mean(coeffs,axis=0)[0]
    stds = np.std(coeffs, axis=0)[0]
    if normalise:
        ave_coeffs = ave_coeffs/stds
        stds = stds/stds
        #stds = np.std(coeffs, axis=0)[0]
    if errors == 'std':
        error_bars = stds
    elif errors == 'p95':
        # this is an approximation assuming a gaussian distribution in your coeffs
        error_bars = 2*stds
    else:
        printf(f'Error: errors setting {errors} not known, chose std or p95')
    
    return ave_coeffs, error_bars


def coeff_plot(coeffs, labels, errors='std',normalise=False):
    """Coefficient plot
    set error to 'std' for standard deviation
    set error to 'p95' for 95th percentile (
    approximated by 2*std)"""
    # get values
    ave_coeffs, error_bars = calc_ave_coeffs_and_errors(
        coeffs, 
        labels, 
        errors=errors,
        normalise=normalise)
    # plot values
    x_points = [x for x in range(len(ave_coeffs))]
    f = plt.figure()
    f.set_figwidth(16)
    f.set_figheight(9)
    print(f"Input_selector was: {x_points}")
    print(f"Average coefficients are: {ave_coeffs}")
    #print(errors)
    print(f"Coefficient labels are: {labels}")
    plt.bar(x_points,
            ave_coeffs,
            yerr=error_bars, capsize=20)
    plt.xticks(x_points,labels)
    return 

def calulate_R2_and_Q2_for_models(inputs, 
                           responses, 
                           input_selector=None, 
                           response_selector=None,
                           fit_intercept=True,
                           use_scaled_inputs=False,
                           do_scaling_here=False,
                           drop_duplicates='average',
                           do_plot=True,
                           do_r2=True,
                           verbose = True):    
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
    #if use_scaled_inputs:
    #    inputs = orthogonal_scaling(inputs)
    # finds out which columns we're going to use
    input_column_list = [x for x in inputs.columns]
    response_column_list = [x for x in responses.columns]
    if verbose:
        print(f"Input terms are {input_column_list}")
        print(f"Input Responses are {response_column_list}\n")
    # make a linear regression model
    if response_selector == None:
        # do all columns
        res_col_num_list = [x for x in range(len(responses.columns))]
    else:
        res_col_num_list = range(len(response_selector))
    if input_selector == None:
        #saturated model - do all columns
        input_selector = [x for x in inp_col_num_list]
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
        edited_input_data=inputs.iloc[:, input_selector]
        selected_input_terms = [x for x in edited_input_data.columns]
        print("Selected input terms:\t{}".format(selected_input_terms))
        # makes a new model
        temp_tuple = calc_averaged_model(
        edited_input_data, 
        this_model_responses, 
        key=response_key,
        drop_duplicates=drop_duplicates,
        fit_intercept=fit_intercept,
        use_scaled_inputs=use_scaled_inputs,
        do_scaling_here=do_scaling_here)        
        new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
        # we fit it as this is hte easiest way to set up the new model correctly
        new_model.fit(edited_input_data, this_model_responses)
        # we overwrite the coefficients with the averaged model
        if len(res_col_num_list)==1:
            coefficient_list = new_model.coef_[0]
        else:
            coefficient_list = new_model.coef_[res_col_num]
        #selected_coefficient_list = coefficient_list[input_selector]
        #print("Coefficients:\t{}".format(selected_coefficient_list))
        new_model.coef_ = coefficient_list
        # now get the R2 value
        R2 = new_model.score(edited_input_data, this_model_responses)
        if verbose:
            print("Response {} R2 is {:.3}".format(response_key, R2))
            print("Input selector was {}".format(input_selector))
        if not do_r2:
            plt.clf()
        if do_plot:
            coeff_plot(coeffs, 
                       labels=selected_input_terms, 
                       errors='p95',
                       normalise=True)
        #print(averaged_coeffs)
        #scaled_coeffs = orthogonal_scaling(coefficient_list)
        #plt.bar([x for x in range(len(scaled_coeffs))],scaled_coeffs)
        

    return new_model, R2, temp_tuple, selected_input_terms

def tune_model(inputs, 
               responses, 
               input_selector=None, 
               response_selector=None,
               fit_intercept=True,
               use_scaled_inputs=False,
               do_scaling_here=False,
               drop_duplicates='average',
               do_plot=True,
               do_r2=True,
               verbose = True
              ):
    """Wrapper to calulate_R2_and_Q2_for_models to make life easy
       It does both scaled and unscaled models
       assumes you want an unscaled model for ease of plotting
       and a scaled model coefficients for ease of choosing"""#
    
    ## scaled model, use this for picking your coefficients
    this_model, R2, temp_tuple, selected_input_terms = calulate_R2_and_Q2_for_models(
                        inputs, 
                        responses[['Profit']], 
                        input_selector=input_selector, 
                        response_selector=response_selector,
                        use_scaled_inputs=True,
                        drop_duplicates='No',
                        do_scaling_here=True)
    scaled_model, predictions, ground_truth, coeffs, R2s, R2, Q2= temp_tuple
    
    ## unscaled model, use this for picking your coefficients
    #this_model, R2, temp_tuple, selected_input_terms = calulate_R2_and_Q2_for_models(
    #                    inputs, 
    #                    responses[['Profit']], 
    #                    input_selector=input_selector, 
    #                    response_selector=response_selector,
    #                    use_scaled_inputs=False,
    #                    drop_duplicates='No',
    #                    do_scaling_here=False,
    #                    do_plot=False,
    #                    verbose=False)
    #unscaled_model, predictions, ground_truth, coeffs, R2s, R2, Q2= temp_tuple
    unscaled_model = []

    return scaled_model, R2, temp_tuple, selected_input_terms

def autotune_model(
        inputs,
        responses,
        source_list,
        response_selector=[0],
        use_scaled_inputs=True,
        do_scaling_here=True,
        drop_duplicates='average',
        errors='p95',
        normalise=True,
        do_hierarchical=True,
        remove_significant = False,
        verbose=False):
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
        print(f'Source list is {source_list}')
    input_selector=[i for i in range(len(sat_inputs.columns))]
    output_indices = input_selector
    # global list of all column names.
    input_terms=list(sat_inputs.columns)
    if verbose:
        print('numbers\tnames')
        for i,v in enumerate(input_terms):
            print(f"{i}\t{v}")
    have_removed = True
    R2_over_opt = []
    Q2_over_opt = []
    n_terms_over_opt = []
    terms = []
    
    while have_removed:
        selected_input_indices = output_indices
    
        #print(selected_input_indices)
        #print(sat_inputs)
        if len(selected_input_indices) == 0:
            break
        this_model, R2, temp_tuple, selected_input_terms = calulate_R2_and_Q2_for_models(
                        sat_inputs, 
                        responses, 
                        input_selector=selected_input_indices, 
                        response_selector=[0],
                        use_scaled_inputs=True,
                        do_scaling_here=True,
                        drop_duplicates=drop_duplicates,
                        do_r2=False,
                        verbose=False)
        new_model, predictions, ground_truth, coeffs, R2s, R2, Q2 = temp_tuple
    
        R2_over_opt.append(R2)
        Q2_over_opt.append(Q2)

        
        # cell 2
        #print("Cell 2:")
        print(f"Selected terms {selected_input_terms}")
        print(f"Source List: {source_list}")
        # build a dictionary mapping from input term to the set of derived term's indices.
        # Note that we are only caring about indices still in selected_input_indices (I.e. ones we have not thrown out!)
        dependency_dict={}
        for i in selected_input_indices:
            dependency_dict[i]=set()
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
                        except:
                            if do_hierarchical:
                                print(f"Error: Heirarchical model missing lower level terms!!!!")
        print(dependency_dict)
        # Handy shortcut - since the empty set is considered false,
        # we can just test dependency_dict[some_term] to see if there are still dependents.
        
        #cell 3
        enforce_hierarchical_model = do_hierarchical
        # whether to enforce hierarchy over terms (i.e. a lower order term must be present for a higher order one)

        ave_coeffs, error_bars=calc_ave_coeffs_and_errors(
            coeffs=coeffs, 
            labels=selected_input_terms, 
            errors='p95',
            normalise=True)
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
                    print(f'{i}:\t{input_terms[i]}\t{source_list[i]}')
                insignificant_terms.append((selected_input_indices[i], abs(ave_coeffs[i])))
                #diffs.append(abs(ave_coeffs[i]) - abs(error_bars[i]))
                
        # for removing smallest significant terms
        if insignificant_terms == [] and remove_significant:
            for i in range(len(ave_coeffs)):
               # print("{:.2}- {:.2}", ave_coeffs[i], error_bars[i])
                print(f'{i}:\t{input_terms[i]}\t{source_list[i]}')
                insignificant_terms.append((selected_input_indices[i], abs(ave_coeffs[i])))
                #diffs.append(abs(ave_coeffs[i]) - abs(error_bars[i]))

        # Now sort in order of ave_coeff
        insignificant_terms = sorted(insignificant_terms, key=lambda x: x[1])
        print(insignificant_terms)

        # Now find the first term we can remove (if any)
        have_removed = False

        for idx, error_value in insignificant_terms:
            # If it has dependents, and you're doing an heirarchical model skip it
            if do_hierarchical==True:
                if dependency_dict[idx]:
                    continue
            print(f"removing term {input_terms[idx]} ({idx}) with error {error_value}")
            output_indices.remove(idx)
            have_removed = True
            break

        print(f"output_indices are {output_indices}")
        terms.append(output_indices)
    return output_indices, new_model, predictions, ground_truth, coeffs, R2s, R2, Q2, R2_over_opt, Q2_over_opt, n_terms_over_opt, terms




def map_chemical_space(
    unscaled_model,
    x_key,
    y_key,
    c_key,
    x_limits,
    y_limits,
    constant,
    n_points,
    my_function):

    min_x = x_limits[0]
    max_x = x_limits[1]
    min_y = y_limits[0]
    max_y = y_limits[1]

    x = np.linspace(min_x, max_x, n_points)
    y = np.linspace(min_y, max_y, n_points)
    c = np.linspace(constant, constant, n_points)

    def fn(x, y, c, unscaled_model, my_function):
        df_1 = pd.DataFrame()
        df_1[x_key] = x.reshape(-1)
        df_1[y_key] = y.reshape(-1)
        df_1[c_key] = c
        # here we add the extra terms
        df_1 = my_function(df_1)
        #print(df_1)
        z = unscaled_model.predict(df_1)
        return z

    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y, constant, unscaled_model, my_function)
    #print(Z.shape)
    #print(Z)
    Z=Z.reshape(n_points,n_points)
    return X, Y, Z


def map_chemical_space_new(
        unscaled_model,
        x_key,
        y_key,
        c_key,
        x_limits,
        y_limits,
        constant,
        n_points,
        my_function,
        source_list=[]):
    min_x = x_limits[0]
    max_x = x_limits[1]
    min_y = y_limits[0]
    max_y = y_limits[1]

    x = np.linspace(min_x, max_x, n_points)
    y = np.linspace(min_y, max_y, n_points)
    c = np.linspace(constant, constant, n_points)

    def fn(x, y, c, unscaled_model, my_function):
        df_1 = pd.DataFrame()
        df_1[x_key] = x.reshape(-1)
        df_1[y_key] = y.reshape(-1)
        df_1[c_key] = c
        # here we add the extra terms
        df_1 = my_function(df_1)
        sat_inputs=add_higher_order_terms(df_1,
                               add_squares=True,
                               add_interactions=True,
                               column_list=source_list,
                               verbose=True)

        predict_from_model(model,
                           inputs,
                           input_selector)
        # print(df_1)
        z = unscaled_model.predict(df_1)
        return z

    X, Y = np.meshgrid(x, y)
    Z = fn(X, Y, constant, unscaled_model, my_function)
    # print(Z.shape)
    # print(Z)
    Z = Z.reshape(n_points, n_points)
    return X, Y, Z

def my_function(df_1):
    
    pass
    return df_1

def four_D_contour_plot(
    unscaled_model,
    x_key,
    y_key,
    c_key,
    x_limits, 
    y_limits, 
    constants, 
    n_points,
    my_function,
    fig_label='',
    input_selector=[],
    x_label='',
    y_label='',
    constant_label='',
    z_label = '',
    cmap='jet',
    num_of_z_levels=9,
    z_limits=[]):
    """This could be improved to take any number of data
    1. unscaled_model: the model you just trained
    2. x_key: name in the dataframe for the input to go on the x axis
    3. y_key: name in the dataframe for the input to go on the y axis
    4. c_key: name in the dataframe for the input to be the constant for each plot (i.e. equivalents of pyrollidine)
    5. x_limits: limits of the x axis: min and max time values
    6. y_limits: limits of the y axis: min and max temperatures
    7. constants: values of pyrollidine to keep constant for the 3 plots
    8. n_points: how many points in the x and y direction to use to build the map
    9. my_function: a little function to add higher order terms if the model requires it
    10. fig_label: label for the overall figure
    11. x_label: label for x axes
    12. y_label: label for y axis
    13. constant_label: label for top of subplots
    14: z_label: label for the heatbar
    15: cmap: colourmap for the plot (yes you can change it, do not spend hours playing around with the colourscheme!)
    16: num_of_z_levels: number of levels for the contours. You will want one more than you think you do
    17: z_limits: limits for the yield, i.e. minimum and maximum. """




    X_1,Y_1,Z_1 = map_chemical_space(
        unscaled_model=unscaled_model,
        x_key = x_key,
        y_key = y_key,
        c_key = c_key,
        x_limits=x_limits, 
        y_limits=y_limits, 
        constant=constants[0], 
        n_points=n_points,
        my_function=my_function)
    X_2,Y_2,Z_2 = map_chemical_space(
        unscaled_model=unscaled_model,
        x_key = x_key,
        y_key = y_key,
        c_key = c_key,
        x_limits=x_limits, 
        y_limits=y_limits, 
        constant=constants[1], 
        n_points=n_points,
        my_function=my_function)
    X_3,Y_3,Z_3 = map_chemical_space(
        unscaled_model=unscaled_model,
        x_key = x_key,
        y_key = y_key,
        c_key = c_key,
        x_limits=x_limits, 
        y_limits=y_limits, 
        constant=constants[2], 
        n_points=n_points,
        my_function=my_function)

    if x_label == '':
        x_label = x_key
    if y_label == '':
        y_label = y_key
    if z_label =='':
        z_label=fig_label
    if constant_label == '':
        constant_label = c_key
        

    if z_limits == []:
        z_min = np.min([np.min(Z_1), np.min(Z_2), np.min(Z_3)])
        z_max = np.max([np.max(Z_1), np.max(Z_2), np.max(Z_3)])
    else:
        z_min = z_limits[0]
        z_max = z_limits[1]


    plt.figure(figsize=(20, 12))
    num_of_levels = 9
    levels = np.linspace(z_min, z_max, num_of_z_levels)
    #levels = np.linspace(20, 100, num_of_levels)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(fig_label)
    egg=ax1.contourf(X_1, Y_1, Z_1, 
                     num_of_levels, levels=levels, cmap=cmap)

    egg1=ax1.contour(X_1, Y_1, Z_1,  
                     num_of_levels, levels=levels, colors = 'black')
    if np.max(levels) >10:
        plt.clabel(egg1, fontsize=10, inline=1,fmt = '%1.0f')
    else:
        plt.clabel(egg1, fontsize=10, inline=1, fmt='%1.2f')
    #egg.xlabel('egg')
    ax2.contourf(X_2, Y_2, Z_2, 
                 num_of_levels, levels=levels,cmap=cmap)
    egg2=ax2.contour(X_2, Y_2, Z_2,  num_of_levels, levels=levels,
                     colors = 'black')
    if np.max(levels) >10:
        plt.clabel(egg2, fontsize=10, inline=1,fmt = '%1.0f')
    else:
        plt.clabel(egg2, fontsize=10, inline=1, fmt='%1.2f')
    ax3.contourf(X_3, Y_3, Z_3, 
                 num_of_levels, levels=levels,cmap=cmap)
    egg3=ax3.contour(X_3, Y_3, Z_3,  
                     num_of_levels, levels=levels, colors = 'black')
    if np.max(levels) >10:
        plt.clabel(egg3, fontsize=10, inline=1,fmt = '%1.0f')
    else:
        plt.clabel(egg3, fontsize=10, inline=1, fmt='%1.2f')
    # make constant label for subplot title
    ax1.set_title(constant_label + ' = ' + str(constants[0]))
    ax2.set_title(constant_label + ' = ' + str(constants[1]))
    ax3.set_title(constant_label + ' = ' + str(constants[2]))
    # make constant label for subplot title
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    ax3.set_xlabel(x_label)
    
    ax1.set_ylabel(y_label)
    
        #fig.colorbar()
        #ax1.clabel(contours, inline=True, fontsize=12)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(egg, cax=cbar_ax)
    #cbar.set_label(z_label, rotation=270)
    cbar.set_label(z_label, labelpad=0, y=1.10, rotation=0)
 
    return

def add_higher_order_terms(inputs,
                          add_squares=True,
                          add_interactions=True,
                          column_list=[],
                          verbose= True):
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
          
    if add_squares == True:
        if verbose:
            print('Adding square terms:')
        for i in range(len(column_list)):
            source_list.append(i)
            input_name = column_list[i]
            new_name = input_name + '**2'
            if verbose:
                print(new_name)
            sat_inputs[new_name] = inputs[input_name]*inputs[input_name]

            
    if add_interactions == True:
        if verbose:
            print('Adding interaction terms:')
        for i in range(len(column_list)):
            for j in range(i+1,len(column_list)):
                source_list.append([i,j])
                input_name_1 = column_list[i]
                input_name_2 = column_list[j]
                new_name = input_name_1 + '*' + input_name_2
                if verbose:
                    print(new_name)
                sat_inputs[new_name] = inputs[input_name_1]*inputs[input_name_2]

    return sat_inputs, source_list

def plot_training(R2_over_opt,
                  Q2_over_opt,
                  n_terms_over_opt):
    """Plots optimisation correlation coefficient outcomes
    n_terms_over_opt
    R2_over_opt: list of R2 over optimisation
    Q2_over_opt: list of Q2 over optimisation
    n_terms_over_opt: running number of terms 
                  
                  """
    ax = plt.axes()
    x_data = range(len(R2_over_opt))
    plt.plot(x_data, R2_over_opt)
    plt.plot(x_data, Q2_over_opt)
    #plt.set_xlabels(n_terms_over_opt)
    ax.set_xticks(x_data)
    if not n_terms_over_opt == []:
        ax.set_xticklabels(n_terms_over_opt)
    plt.legend(["R$^2$","Q$^2$"])
    plt.ylim([0,1])
    plt.plot([x_data[0],x_data[-1]],[0.5,0.5], '-.')
    plt.xlabel('Number of terms')
    plt.ylabel("Correlation coefficient");
    return ax

def predict_from_model(model,
                        inputs,
                        input_selector):
    """Reorgs the inputs and does a prediction
    model = the model to use
    inputs = the saturated inputs 
    input_selector = the subset of inputs the model is using
    """
    list_of_terms=[inputs.columns[x] for x in input_selector]
    model_inputs = inputs[list_of_terms]
    predictions = model.predict(model_inputs)
    return predictions, list_of_terms
