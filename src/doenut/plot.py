"""
doenut.plot

Provides graph plotting functions for DoeNut
"""

from matplotlib import pyplot as plt
import numpy as np
import doenut


def clear_figure():
    """
    Wrapper for matplotlib clear figure to avoid imports elsewhere
    """
    plt.clf()


def replicate_plot(inputs, responses, key):
    """Plots a replicate plot which shows all experiments
    and identifies the replicates
    inputs:
    responses:
    key: column in responses that you wish to plot"""
    plt.title(key)
    replicate_row_list = doenut.find_replicates(inputs)
    non_replicate_row_list = [
        x for x in range(len(responses)) if x not in replicate_row_list
    ]
    col = responses[key]
    x_data = [x for x in range(len(non_replicate_row_list))]
    y_data = [col.loc[x] for x in non_replicate_row_list]
    plt.plot(x_data, y_data, "o")
    ax = plt.gca()
    ax.set_xticks(x_data)
    ax.set_xticklabels(non_replicate_row_list)
    plt.xlabel("Experiment No.")
    plt.ylabel("Response")

    x_data = [
        len(non_replicate_row_list) for _ in range(len(replicate_row_list))
    ]
    y_data = [col.loc[x] for x in replicate_row_list]
    plt.plot(x_data, y_data, "or")
    return


def plot_observed_vs_predicted(
    responses, predictions, range_x=None, label="", do_axes_equal=True
):
    """plots a graph duh
    range should be in the form [min_x, max_x]
    else it will take from responses"""
    if range_x is None:
        range_x = []
    plt.plot(responses, predictions, "o")
    min_xy = np.min([np.min(responses), np.min(predictions)])
    max_xy = np.max([np.max(responses), np.max(predictions)])
    if not range_x:
        range_x = [(min_xy // 10) * 10, (max_xy // 10) * 10 + 10]
    plt.plot(range_x, range_x)
    x_label = "Measured " + label
    y_label = "Predicted " + label
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if do_axes_equal:
        plt.gca().set_aspect("equal")
    return


def plot_summary_of_fit_small(R2, Q2):
    """Plots a nice graph of R2 and Q2"""
    my_colors = [
        "green",
        "blue",
        "pink",
        "cyan",
        "red",
        "green",
        "blue",
        "cyan",
        "orange",
        "purple",
    ]
    plt.bar([1, 2], [R2, Q2], color=my_colors)
    plt.grid(True)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.ylabel('Average Price')
    plt.xticks([1, 2], labels=["R2", "Q2"])
    plt.ylim((0, 1))
    plt.grid(axis="x")
    return


def coeff_plot(coeffs, labels, errors="std", normalise=False):
    """Coefficient plot
    set error to 'std' for standard deviation
    set error to 'p95' for 95th percentile (
    approximated by 2*std)"""
    # get values
    ave_coeffs, error_bars = doenut.calc_ave_coeffs_and_errors(
        coeffs, labels, errors=errors, normalise=normalise
    )
    # plot values
    x_points = [x for x in range(len(ave_coeffs))]
    f = plt.figure()
    f.set_figwidth(16)
    f.set_figheight(9)
    print(f"Input_selector was: {x_points}")
    print(f"Average coefficients are: {ave_coeffs}")
    print(f"Errors are {error_bars}")
    # print(errors)
    print(f"Coefficient labels are: {labels}")
    plt.bar(x_points, ave_coeffs, yerr=error_bars, capsize=20)
    plt.xticks(x_points, labels)
    return


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
    fig_label="",
    input_selector=None,
    x_label="",
    y_label="",
    constant_label="",
    z_label="",
    cmap="jet",
    num_of_z_levels=9,
    z_limits=None,
    tidy_subfig_axes=False,
):
    """This could be improved to take any number of data
    1. unscaled_model: the model you just trained
    2. x_key: name in the dataframe for the input to go on the x-axis
    3. y_key: name in the dataframe for the input to go on the y-axis
    4. c_key: name in the dataframe for the input to be the constant for each plot (i.e. equivalents of pyrollidine)
    5. x_limits: limits of the x-axis: min and max time values
    6. y_limits: limits of the y-axis: min and max temperatures
    7. constants: values of pyrollidine to keep constant for the 3 plots
    8. n_points: how many points in the x and y direction to use to build the map
    9. my_function: a little function to add higher order terms if the model requires it
    10. fig_label: label for the overall figure
    11. x_label: label for x-axes
    12. y_label: label for y-axis
    13. constant_label: label for top of subplots
    14: z_label: label for the heatbar
    15: cmap: colourmap for the plot (yes you can change it, do not spend hours playing around with the colourscheme!)
    16: num_of_z_levels: number of levels for the contours. You will want one more than you think you do
    17: z_limits: limits for the yield, i.e. minimum and maximum."""

    if input_selector is None:
        input_selector = []
    if z_limits is None:
        z_limits = []
    X_1, Y_1, Z_1 = doenut.map_chemical_space(
        unscaled_model=unscaled_model,
        x_key=x_key,
        y_key=y_key,
        c_key=c_key,
        x_limits=x_limits,
        y_limits=y_limits,
        constant=constants[0],
        n_points=n_points,
        hook_function=my_function,
    )
    X_2, Y_2, Z_2 = doenut.map_chemical_space(
        unscaled_model=unscaled_model,
        x_key=x_key,
        y_key=y_key,
        c_key=c_key,
        x_limits=x_limits,
        y_limits=y_limits,
        constant=constants[1],
        n_points=n_points,
        hook_function=my_function,
    )
    X_3, Y_3, Z_3 = doenut.map_chemical_space(
        unscaled_model=unscaled_model,
        x_key=x_key,
        y_key=y_key,
        c_key=c_key,
        x_limits=x_limits,
        y_limits=y_limits,
        constant=constants[2],
        n_points=n_points,
        hook_function=my_function,
    )

    if x_label == "":
        x_label = x_key
    if y_label == "":
        y_label = y_key
    if z_label == "":
        z_label = fig_label
    if constant_label == "":
        constant_label = c_key

    if not z_limits:
        z_min = np.min([np.min(Z_1), np.min(Z_2), np.min(Z_3)])
        z_max = np.max([np.max(Z_1), np.max(Z_2), np.max(Z_3)])
    else:
        z_min = z_limits[0]
        z_max = z_limits[1]

    plt.figure(figsize=(20, 12))
    num_of_levels = 9
    levels = np.linspace(z_min, z_max, num_of_z_levels)
    # levels = np.linspace(20, 100, num_of_levels)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(fig_label)
    egg = ax1.contourf(X_1, Y_1, Z_1, num_of_levels, levels=levels, cmap=cmap)

    egg1 = ax1.contour(
        X_1, Y_1, Z_1, num_of_levels, levels=levels, colors="black"
    )
    if np.max(levels) > 10:
        plt.clabel(egg1, fontsize=16, inline=1, fmt="%1.0f")
    else:
        plt.clabel(egg1, fontsize=16, inline=1, fmt="%1.2f")
    # egg.xlabel('egg')
    ax2.contourf(X_2, Y_2, Z_2, num_of_levels, levels=levels, cmap=cmap)
    egg2 = ax2.contour(
        X_2, Y_2, Z_2, num_of_levels, levels=levels, colors="black"
    )
    if np.max(levels) > 10:
        plt.clabel(egg2, fontsize=16, inline=1, fmt="%1.0f")
    else:
        plt.clabel(egg2, fontsize=16, inline=1, fmt="%1.2f")
    ax3.contourf(X_3, Y_3, Z_3, num_of_levels, levels=levels, cmap=cmap)
    egg3 = ax3.contour(
        X_3, Y_3, Z_3, num_of_levels, levels=levels, colors="black"
    )
    if np.max(levels) > 10:
        plt.clabel(egg3, fontsize=16, inline=1, fmt="%1.0f")
    else:
        plt.clabel(egg3, fontsize=16, inline=1, fmt="%1.2f")
    # make constant label for subplot title
    ax1.set_title(constant_label + " = " + str(constants[0]))
    ax2.set_title(constant_label + " = " + str(constants[1]))
    ax3.set_title(constant_label + " = " + str(constants[2]))
    # make constant label for subplot title
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    ax3.set_xlabel(x_label)

    ax1.set_ylabel(y_label)

    if tidy_subfig_axes:
        ax2.yaxis.set_visible(False)
        ax3.yaxis.set_visible(False)

        # fig.colorbar()  # ax1.clabel(contours, inline=True, fontsize=12)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(egg, cax=cbar_ax)
    # cbar.set_label(z_label, rotation=270)
    cbar.set_label(z_label, labelpad=0, y=1.10, rotation=0)

    return


def plot_training(R2_over_opt, Q2_over_opt, n_terms_over_opt):
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
    # plt.set_xlabels(n_terms_over_opt)
    ax.set_xticks(x_data)
    if not n_terms_over_opt == []:
        ax.set_xticklabels(n_terms_over_opt)
    plt.legend(["R$^2$", "Q$^2$"])
    plt.ylim([0, 1])
    plt.plot([x_data[0], x_data[-1]], [0.5, 0.5], "-.")
    plt.xlabel("Number of terms")
    plt.ylabel("Correlation coefficient")
    return ax
