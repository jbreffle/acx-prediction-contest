"""Plotting functions for the 2023 blind mode forecasting project.
"""

# Imports
import math
import statistics
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import beta
from scipy.stats import linregress
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind

from src import process


# Functions
def score_vs_beta_2d(beta_range, score_grid, ax=None):
    if ax is None:
        ax = plt.gca()
    colormap_name = "viridis"  # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    im = ax.imshow(
        score_grid,
        interpolation="nearest",
        origin="lower",
        cmap=mpl.colormaps[colormap_name],
    )
    ax.set_title("MSE vs beta_a and beta_b")
    ax.set_xlabel(r"$\beta_b$")
    ax.set_ylabel(r"$\beta_a$")
    # Correct the tick labels to show the actual values, but only show current number of ticks
    target_ticks = 10
    tick_interval = max(1, int(max(len(beta_range), len(beta_range)) / target_ticks))
    # Set tick locations
    ax.set_xticks(np.arange(0, len(beta_range), tick_interval))
    ax.set_yticks(np.arange(0, len(beta_range), tick_interval))
    # Set tick labels
    ax.set_xticklabels([f"{b:.2f}" for b in beta_range[::tick_interval]], rotation=45)
    ax.set_yticklabels([f"{a:.2f}" for a in beta_range[::tick_interval]])
    # Add annotation of minimum value
    min_idx = np.unravel_index(np.argmin(score_grid), score_grid.shape)
    min_mse_beta = score_grid[min_idx]
    min_beta_a = beta_range[min_idx[0]]
    min_beta_b = beta_range[min_idx[1]]
    ax.annotate(
        rf"  Min Brier: {min_mse_beta:.4f} at $\beta_a$={min_beta_a:.4f}, $\beta_b$={min_beta_b:.4f}",
        xy=(min_idx[1], min_idx[0]),  # x, y coordinates for annotation point
        xytext=(min_idx[1] * 0.0, min_idx[0] * 1.75),  # Adjust text position
        color="black",  # Make sure it is black
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    # Add line at beta_a==beta_b
    ax.plot([0, len(beta_range) - 1], [0, len(beta_range) - 1], color="grey")
    # Add colorbar
    plt.colorbar(im, ax=ax)
    return ax


def score_vs_beta(beta_range, score_vec, ax=None):
    """..."""
    if ax is None:
        ax = plt.gca()
    ax.plot(beta_range, score_vec)
    ax.set_title("Brier score vs beta parameters")
    ax.set_xlabel(r"$\beta_a = \beta_b$")
    ax.set_ylabel("Brier score")
    ax.set_xscale("log")
    # Add annotation to minimum
    min_mse_beta = np.min(score_vec)
    min_beta_a = beta_range[np.argmin(score_vec)]
    ax.annotate(
        rf"Min Brier: {min_mse_beta:.4f} at $\beta_a = \beta_b$={min_beta_a:.4f}",
        xy=(min_beta_a, min_mse_beta),
        xytext=(min_beta_a, min_mse_beta * 1.2),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    return ax


def feature_volcano_plot(r_and_p_values, significance_threshold=0.05, ax=None):
    """Plot a volcano plot of the r and p values of the features."""
    # Color by significance and slope
    color = np.array(
        [
            "red"
            if (x < significance_threshold and y > 0)
            else "green"
            if (x < significance_threshold and y < 0)
            else "grey"
            for x, y in zip(r_and_p_values["p_value"], r_and_p_values["r_value"])
        ]
    )
    marker_size = 20
    if ax is None:
        ax = plt.gca()
    # Plot volcano plot of r vs p values
    ax.scatter(
        r_and_p_values["r_value"],
        -np.log10(r_and_p_values["p_value"]),
        color=color,
        s=marker_size,
    )
    ax.set_title("Feature correlations to brier score, r vs p values")
    ax.set_xlabel("r-value")
    ax.set_ylabel("-log10 p-value")
    # legend: green: significant improvement to score
    #         red: significant decrease to score
    #         grey: not significant
    # Define the legend markers as Line2D objects
    legend_marker_size = marker_size / 4
    green_line = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=legend_marker_size,
        label="Better score",
    )
    red_line = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        markersize=legend_marker_size,
        label="Worse score",
    )
    grey_line = mlines.Line2D(
        [],
        [],
        color="grey",
        marker="o",
        linestyle="None",
        markersize=legend_marker_size,
        label="Not significant",
    )
    plt.legend(
        handles=[green_line, red_line, grey_line],
        loc="upper center",
        framealpha=0.0,
        fancybox=True,
    )
    return ax


def feature_scatter_hist(
    blind_mode_feature_df, brier_score, feature_to_plot, fig=None, **hist_kwargs
):
    """..."""
    # Correlation of feature with brier score
    if fig is None:
        fig = plt.figure(figsize=(6, 4))
    x = blind_mode_feature_df[feature_to_plot]
    y = brier_score
    nan_mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, _ = linregress(x[nan_mask], y[nan_mask])
    # Check if x is continuous, discrete, or binary, excluding nan
    if len(x.dropna().unique()) > 10:
        x_type = "continuous"
    elif len(x.dropna().unique()) == 2:
        x_type = "binary"
    else:
        x_type = "discrete"
    # Scatter feature with brier score
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Plot to each subplot
    match x_type:
        case "continuous":
            ax.scatter(x, y, s=5)
            ax_histx.hist(x, **hist_kwargs)
            ax_histy.hist(y, orientation="horizontal", **hist_kwargs)
        case "binary" | "discrete":
            x_labels = x.dropna().unique()
            # ax.scatter(x, y, s=5)
            ax.violinplot(
                [y[x == x_label] for x_label in x_labels],
                x_labels,
                showmeans=True,
                showmedians=False,
                showextrema=False,
            )
            # Add histograms
            bins = np.arange(x.min() - 0.5, x.max() + 1.5, 1)
            ax_histx.hist(x, bins=bins, edgecolor="black", align="mid", **hist_kwargs)
            ax_histy.hist(y, orientation="horizontal", **hist_kwargs)
        case _:
            raise ValueError("Invalid x_type")
    if x_type == "binary":
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
    # Label axes
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel("Brier score")
    # create x values that extend beyond the data
    x_to_plot = np.array([x.min() - 0.0, x.max() + 0.0])
    ax.plot(
        x_to_plot,
        intercept + slope * x_to_plot,
        "r-",
    )
    ax.text(
        0.025,
        0.975,
        f"R={r_value:.2f}, p={p_value:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    ax_histx.set_title(f"Brier score vs {feature_to_plot}")
    return ax, ax_histx, ax_histy


def feature_scatter(blind_mode_feature_df, brier_score, feature_to_plot, ax=None):
    """..."""
    # Correlation of feature with brier score
    x = blind_mode_feature_df[feature_to_plot]
    y = brier_score
    nan_mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, _ = linregress(x[nan_mask], y[nan_mask])
    # Scatter feature with brier score
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.scatter(blind_mode_feature_df[feature_to_plot], brier_score, s=5)
    ax.plot(
        blind_mode_feature_df[feature_to_plot],
        intercept + slope * blind_mode_feature_df[feature_to_plot],
        "r-",
    )
    ax.text(
        0.025,
        0.975,
        f"R={r_value:.2f}, p={p_value:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    ax.set_title(f"Brier score vs {feature_to_plot}")
    ax.set_xlabel(feature_to_plot)
    ax.set_ylabel("Brier score")
    return fig


def group_final_brier_violin_plot(
    blind_mode_df, blind_mode_final_brier, print_stats=False
):
    """Plot a violin plot of the final brier scores split by group."""
    # Get the group names
    analysis_category = ["All", "ForecastingExperience", "Superforecaster", "LessWrong"]
    briers_all = blind_mode_final_brier
    briers_fe = blind_mode_final_brier[blind_mode_df["ForecastingExperience"] == "Yes"]
    briers_sf = blind_mode_final_brier[blind_mode_df["Superforecaster"] == "Yes"]
    briers_lw = blind_mode_final_brier[blind_mode_df["LessWrong"] == "Yes"]
    # Make a violin plot of the brier scores split by group
    fig = plt.figure(figsize=(8, 3))
    plt.violinplot([briers_all, briers_fe, briers_sf, briers_lw], showmeans=True)
    plt.xticks([1, 2, 3, 4], analysis_category)
    plt.ylabel("Brier score")
    plt.title("Brier score by group")
    # Calculate stastistics comparing each groups scores
    if print_stats:
        print("All vs. ForecastingExperience:", ttest_ind(briers_all, briers_fe))
        print("All vs. Superforecaster:", ttest_ind(briers_all, briers_sf))
        print("All vs. LessWrong:", ttest_ind(briers_all, briers_lw))
    return fig


def my_score_time_series(my_pred_df, market_hist_df, transient=0):
    """Calcualtes and plots the MSE and Brier score time series for all questions."""
    # TODO make score_type an argument and plot single time series
    # Calculate both scores
    my_mse = process.get_score_df(my_pred_df, market_hist_df, score_type="mse")
    my_brier = process.get_score_df(my_pred_df, market_hist_df, score_type="brier")
    # Plot overall MSE
    fig_1 = plt.figure(figsize=(8, 2))
    plt.plot(
        [datetime.datetime.fromtimestamp(time / 1000) for time in my_mse["time"]][
            transient:
        ],
        my_mse[transient:][0],
    )
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.title("MSE for all questions")
    # Plot overall Brier score
    fig_2 = plt.figure(figsize=(8, 2))
    plt.plot(
        [datetime.datetime.fromtimestamp(time / 1000) for time in my_brier["time"]][
            transient:
        ],
        my_brier[transient:][0],
    )
    plt.xlabel("Time")
    plt.ylabel("Brier score")
    plt.title("Brier score for all questions")
    return fig_1, fig_2


def question_market_time_series(q_to_plot, q_to_plot_prediction, market_hist_df):
    """Calculates and plots Market history, RMSE, and Brier score for a question.

    Note, 0.5 is a weird case for the brier score. It is constant for either outcome.
    """
    # TODO plot type an argument and plot single time series
    q_to_plot = f"Q{q_to_plot}"
    time_vec = [
        datetime.datetime.fromtimestamp(time / 1000) for time in market_hist_df["time"]
    ]
    question_most_likely_outcome = np.array(
        [round(prob) for prob in market_hist_df[q_to_plot]]
    )
    rmse_vec = np.sqrt((q_to_plot_prediction - market_hist_df[q_to_plot]) ** 2)
    brier_score_vec = np.sqrt(
        (q_to_plot_prediction - question_most_likely_outcome) ** 2
    )
    # Plot market history of q_to_plot
    fig_1 = plt.figure(figsize=(8, 2))
    plt.plot(time_vec, market_hist_df[q_to_plot])
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title("Probability history of market: " + q_to_plot)
    plt.ylim([0, 1])
    # Plot mse of q_to_plot relative to q_to_plot_prediction
    fig_2 = plt.figure(figsize=(8, 2))
    plt.plot(time_vec, rmse_vec)
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.title("RMSE history of market: " + q_to_plot)
    plt.ylim([0, 1])
    # Plot Brier score of q_to_plot relative to q_to_plot_prediction
    fig_3 = plt.figure(figsize=(8, 2))
    plt.plot(time_vec, brier_score_vec)
    plt.xlabel("Time")
    plt.ylabel("Brier score")
    plt.title("Brier score history of market: " + q_to_plot)
    plt.ylim([0, 1])
    return fig_1, fig_2, fig_3


def empirical_abs_diff_cdf_by_class(df, calc_statistics=False):
    """Given a dataframe with predictions and class labels,
    plot the empirical CDF of the predictions.
    """
    classes = df["class"].unique()
    class_colors = {"all": "#1f77b4", "FE": "#ff7f0e", "SF": "#2ca02c", "LW": "#d62728"}
    bins = np.linspace(0, 50, 100)
    fig = plt.figure(figsize=(10, 4))
    for class_name in classes:
        hist_n, _, _ = plt.hist(
            abs(df.loc[df["class"] == class_name]["prediction"] - 50),
            histtype="step",
            density=True,
            cumulative=True,
            label=class_name,
            bins=bins,
            color=class_colors[class_name],
        )
        if calc_statistics:
            # Perform linear regression fitted to the histogram bars
            #  and plot line of best fit
            # Then print the slope and p-value of a KS test
            #
            # We want to find the slope when the intercept is fixed to zero,
            # so we can't use we scipy.stats.linregress
            x = bins[:-1] + np.diff(bins) / 2
            y = hist_n
            x = x[:, np.newaxis]
            slope, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            intercept = 0

            x_to_plot = np.array([0, 50])
            plt.plot(
                x_to_plot,
                intercept + slope * x_to_plot,
                label=f"{class_name} fit",
                alpha=0.5,
                color=class_colors[class_name],
            )
            print(
                class_name,
                "vs all:",
                ks_2samp(
                    df.loc[df["class"] == class_name]["prediction"],
                    df.loc[df["class"] == "all"]["prediction"],
                ),
            )
            print(
                class_name,
                "mean:",
                np.mean(df.loc[df["class"] == class_name]["prediction"]).round(3),
            )
            print(class_name, "slope:", slope)
    plt.legend(fancybox=True, loc="upper left")
    plt.title("Distribution of predictions")
    plt.xlabel("Deviation of predictions from 50%")
    plt.ylabel("Cumulative probability")
    return fig


def plot_frac_ne_bootstrap_distributions(fractions_dist, df, ne):
    """
    Plots the bootstrap distributions.

    Parameters:
    - fractions_dist: Dictionary of bootstrap distributions for each class.
    - df: DataFrame containing the predictions and class labels.
    - ne: List of values to be excluded in the calculation.
    """
    # Select uniforms bins compatible with all data in fractions_dist
    bins = np.linspace(
        min([min(x) for x in fractions_dist.values()]),
        max([max(x) for x in fractions_dist.values()]),
        100,
    )
    fig = plt.figure(figsize=(10, 4))
    colors = {"all": "#1f77b4", "FE": "#ff7f0e", "SF": "#2ca02c", "LW": "#d62728"}
    for cls, color in colors.items():
        plt.hist(fractions_dist[cls], bins=bins, color=color, alpha=0.5, label=cls)
        actual_frac = len(
            [x for x in df[df["class"] == cls]["prediction"] if x not in ne]
        ) / len(df[df["class"] == cls])
        plt.axvline(actual_frac, color=color)
        plt.axvline(
            np.percentile(fractions_dist[cls], 2.5), color=color, linestyle="dashed"
        )
        plt.axvline(
            np.percentile(fractions_dist[cls], 97.5), color=color, linestyle="dashed"
        )
    plt.legend(fancybox=True)
    plt.title("Bootstrap distributions by predictor")
    plt.xlabel("Fraction of predictions that are not equal to elements in ne")
    plt.ylabel("Count")
    return fig


def all_predictions_deviation_from_50_histogram(abs_diff_df):
    """
    Plots histograms of absolute differences for different classes in the DataFrame.

    Parameters:
    - abs_diff_df: DataFrame containing the absolute differences from 50
    - classes: List of classes to include in the histogram.
    """
    classes = abs_diff_df["class"].unique()
    fig = plt.figure(figsize=(10, 4))
    for class_name in classes:
        plt.hist(
            abs_diff_df.loc[abs_diff_df["class"] == class_name]["prediction"],
            histtype="step",
            stacked=True,
            fill=False,
            density=True,
            label=class_name,
        )
    plt.legend(fancybox=True, loc="upper left")
    plt.title("Distribution of predictions")
    plt.xlabel("Deviation of predictions from 50%")
    plt.ylabel("Density of predictions")
    return fig


def all_prediction_hist_by_group(df, classes=None, bins=None):
    """
    Plots histograms for different classes in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - classes: List of classes to include in the histogram.
    - bins: Bin vector for the histogram.
    """
    if bins is None:
        bins = list(range(0, 100))
    if classes is None:
        classes = ["all", "FE", "SF", "LW"]
    fig = plt.figure(figsize=(10, 4))
    for class_name in classes:
        plt.hist(
            df.loc[df["class"] == class_name]["prediction"],
            histtype="step",
            bins=bins,
            stacked=True,
            fill=False,
            density=True,
            label=f"x_{class_name}",
        )
    plt.xlabel("Prediction (%)")
    plt.ylabel("Probability")
    plt.legend(fancybox=True, bbox_to_anchor=(1.04, 1), loc="upper left")
    return fig


def sim_outcome_prct_dist(my_score_percentiles, print_stats=False, n_bins=None):
    """..."""
    if n_bins is None:
        n_bins = round(math.sqrt(len(my_score_percentiles)))
    fig = plt.figure(figsize=(6, 3))
    plt.hist(my_score_percentiles, bins=n_bins, density=True)
    plt.xlabel("My Brier score percentile")
    plt.ylabel("Probability density")
    if print_stats:
        print(
            "Mean Brier score percentile: ",
            round(statistics.mean(my_score_percentiles), 1),
        )
    return fig


def sim_outcome_dist(
    my_brier_scores, base_brier_scores, print_stats=False, n_bins=None
):
    """..."""
    if n_bins is None:
        n_bins = round(math.sqrt(len(base_brier_scores)))
    fig = plt.figure(figsize=(6, 3))
    plt.hist(
        base_brier_scores,
        bins=n_bins,
        histtype="step",
        stacked=True,
        fill=False,
        density=True,
    )
    plt.hist(
        my_brier_scores,
        bins=n_bins,
        histtype="step",
        stacked=True,
        fill=False,
        density=True,
    )
    plt.legend(["SF mean prediction", "My predictions"], fancybox=True)
    plt.xlabel("Brier score")
    plt.ylabel("Probability density")
    # Print statistics
    if print_stats:
        print(
            "Mean Brier score for SF predictions: ",
            round(statistics.mean(base_brier_scores), 3),
        )
        print(
            "Mean Brier score for my predictions: ",
            round(statistics.mean(my_brier_scores), 3),
        )
    return fig


def beta_transform(original_predictions, a=1 / 7, b=1 / 7):
    """Apply a beta transformation to a set of predictions"""
    transformed_predictions = beta.ppf(original_predictions, a, b)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axline((0, 0), slope=1, linestyle="dashed", color="black")
    plt.scatter(original_predictions, transformed_predictions)
    plt.title("Beta transformed predictions")
    plt.xlabel("Original predictions")
    plt.ylabel("Transformed predictions")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return fig


def demo_pred_hist(data, q_num, print_stats=False):
    """Plot overlapping histograms of a question's predictions split by experience"""
    # Set up data
    q_str = f"@{q_num}."
    ests_fe = data.loc[data["ForecastingExperience"].values == "Yes"]
    ests_sf = data.loc[data["Superforecaster"].values == "Yes"]
    ests_lw = data.loc[data["LessWrong"].values == "Yes"]
    x_all = data.filter(like=q_str, axis=1).squeeze().dropna()
    x_fe = ests_fe.filter(like=q_str, axis=1).squeeze().dropna()
    x_sf = ests_sf.filter(like=q_str, axis=1).squeeze().dropna()
    x_lw = ests_lw.filter(like=q_str, axis=1).squeeze().dropna()
    # Print statistics
    print("Question: ", x_all.name)
    if print_stats:
        print("All participant median: ", x_all.median())
        print("ForecastingExperience median: ", x_fe.median())
        print("Superforecaster median: ", x_sf.median())
        print("LessWrong median: ", x_lw.median())
    # Plot figure
    fig = plt.figure(figsize=(3, 3))
    plt.hist([x_all], histtype="step", stacked=True, fill=False, density=True)
    plt.hist([x_fe], histtype="step", stacked=True, fill=False, density=True)
    plt.hist([x_sf], histtype="step", stacked=True, fill=False, density=True)
    plt.hist([x_lw], histtype="step", stacked=True, fill=False, density=True)
    plt.xlabel("Prediction (%)")
    plt.ylabel("Probability")
    plt.legend(
        ["All", "FE", "SF", "LW"],
        fancybox=True,
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
    )
    return fig


def demo_pred_scatter(
    data, demo_attribue, statistic="mean", fit_line=True, print_stats=False
):
    """Scatter plot of mean prediction split by experience"""
    match statistic:
        case "mean":
            x = (
                data.loc[data[demo_attribue].values == "No"]
                .filter(like="@")
                .mean(numeric_only=True)
            )
            y = (
                data.loc[data[demo_attribue].values == "Yes"]
                .filter(like="@")
                .mean(numeric_only=True)
            )
            x_summary = x.mean()
            y_summary = y.mean()
        case "median":
            x = (
                data.loc[data[demo_attribue].values == "No"]
                .filter(like="@")
                .median(numeric_only=True)
            )
            y = (
                data.loc[data[demo_attribue].values == "Yes"]
                .filter(like="@")
                .median(numeric_only=True)
            )
            x_summary = x.median()
            y_summary = y.median()
        case _:
            print("Invalid statistic")
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axline((0, 0), slope=1, linestyle="dashed", color="black", alpha=0.5)
    plt.scatter(x, y)
    plt.scatter(x_summary, y_summary, color="red", marker="o")
    plt.title("Prediction " + statistic + "s: " + demo_attribue)
    plt.xlabel("No")
    plt.ylabel("Yes")
    if fit_line:
        plt.autoscale(False)
        slope, intercept = linregress(x, y)[0:2]
        p_value = linregress(x, y - x)[3]
        x_to_plot = np.array([0, 100])
        plt.plot(
            x_to_plot,
            intercept + slope * x_to_plot,
            "r",
            label="fitted line",
            alpha=0.5,
        )
        # Calculate the angle in degrees of the fitted line
        angle = math.degrees(math.atan(slope))
        plt.text(0.5, 0.05, "∠ = " + str(round(angle, 1)) + "°", transform=ax.transAxes)
        plt.text(0.5, 0.1, "p = " + str(round(p_value, 4)), transform=ax.transAxes)
    if print_stats:
        print(
            '"Yeses" have average predictions',
            (y_summary - x_summary).round(2),
            'greater than "nos"',
        )
        print(y_summary.round(2), "vs", x_summary.round(2))
    return fig


def pred_hist(data, q_num, ax=None, print_stats=False, **hist_kwargs):
    """Plots a histogram of all predictions for the q_num-th question

    If print_stats=True, is prints statistics of the prediction distribution
    """
    if ax is None:
        ax = plt.gca()
    q_str = f"@{q_num}."
    y = data.filter(like=q_str, axis=1)
    x = y.squeeze()
    plt.hist(x)
    plt.title(x.name, **hist_kwargs)
    plt.xlabel("Prediction (%)")
    plt.ylabel("Count")
    if print_stats:
        print("Median: ", x.median())
        print("Mean:   ", x.mean().round(2))
        print("Std:    ", x.std().round(2))
    return ax
