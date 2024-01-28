# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

from src import util
from src import process
import Home


# Functions for this page
@st.cache_data
def create_frac_ne_bootstrap_distributions(fractions_dist, n_bins=None):
    """
    Plots the bootstrap distributions.

    Parameters:
    - fractions_dist: Dictionary of bootstrap distributions for each class.
    - df: DataFrame containing the predictions and class labels.
    - ne: List of values to be excluded in the calculation.
    """
    # Select uniforms bins compatible with all data in fractions_dist
    # twice the square root of the number of observations in all of the data
    if n_bins is None:
        n_bins = (
            (np.sqrt(sum([len(x) for x in fractions_dist.values()])))
            .round(0)
            .astype(int)
        )
    bin_range = min([min(x) for x in fractions_dist.values()]), max(
        [max(x) for x in fractions_dist.values()]
    )
    bins = np.linspace(bin_range[0], bin_range[1], n_bins)
    bin_step = bins[1] - bins[0]
    # All
    histogram_counts, bin_edges = np.histogram(
        fractions_dist["all"], bins=n_bins, range=bin_range, density=False
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_all = pd.DataFrame({"bin_centers": bin_centers, "counts": histogram_counts})
    # FE
    histogram_counts, bin_edges = np.histogram(
        fractions_dist["FE"], bins=n_bins, range=bin_range, density=False
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_FE = pd.DataFrame({"bin_centers": bin_centers, "counts": histogram_counts})
    # LW
    histogram_counts, bin_edges = np.histogram(
        fractions_dist["LW"], bins=n_bins, range=bin_range, density=False
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_LW = pd.DataFrame({"bin_centers": bin_centers, "counts": histogram_counts})
    # SF
    histogram_counts, bin_edges = np.histogram(
        fractions_dist["SF"], bins=n_bins, range=bin_range, density=False
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_SF = pd.DataFrame({"bin_centers": bin_centers, "counts": histogram_counts})
    # Combine data for plotting
    source = pd.DataFrame(
        {
            "Experience": np.concatenate(
                (
                    np.repeat("All participants", len(hist_all)),
                    np.repeat("Forcasting experience", len(hist_FE)),
                    np.repeat("LessWrong member", len(hist_LW)),
                    np.repeat("Super forecaster", len(hist_SF)),
                ),
            ),
            "bin_centers": np.concatenate(
                (
                    hist_all["bin_centers"],
                    hist_FE["bin_centers"],
                    hist_LW["bin_centers"],
                    hist_SF["bin_centers"],
                )
            ),
            "counts": np.concatenate(
                (
                    hist_all["counts"],
                    hist_FE["counts"],
                    hist_LW["counts"],
                    hist_SF["counts"],
                )
            ),
        }
    )
    # Create the Altair chart
    chart = (
        alt.Chart(source, height=400)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(
                "bin_centers:Q",
                bin=alt.Bin(maxbins=n_bins, extent=bin_range, step=bin_step),
                title="Fraction of predictions that are not equal to elements in ne",
            ),
            y=alt.Y("counts:Q", title="Count (bootstrap iterations)", stack=False),
            color=alt.Color(
                "Experience",
                scale=alt.Scale(range=["white", "yellow", "orange", "red"]),
                legend=alt.Legend(direction="horizontal", orient="top"),
            ),
        )
    )
    return chart


@st.cache_data
def display_group_metrics(
    ests, blind_mode_df, selected_question_number, statistic="mean"
):
    group_stats = calculate_group_statistics(ests, blind_mode_df, statistic=statistic)
    question_group_stats = group_stats.iloc[selected_question_number - 1]
    question_mean_all = question_group_stats.loc["All_Participants"]
    question_mean_FE = question_group_stats.loc["Forcasting_experience"]
    question_mean_LW = question_group_stats.loc["LessWrong_member"]
    question_mean_SF = question_group_stats.loc["Super forecaster"]
    metric_columns = st.columns([0.4, 1, 1, 1, 1, 0.001])
    with metric_columns[1]:
        st.metric("All participants", question_mean_all.round(2))
    with metric_columns[2]:
        st.metric(
            "Forecasting experience",
            question_mean_FE.round(2),
            delta=(question_mean_FE - question_mean_all).round(2),
        )
    with metric_columns[3]:
        st.metric(
            "LessWrong member",
            question_mean_LW.round(2),
            delta=(question_mean_LW - question_mean_all).round(2),
        )
    with metric_columns[4]:
        st.metric(
            "Super forecaster",
            question_mean_SF.round(2),
            delta=(question_mean_SF - question_mean_all).round(2),
        )
    return


@st.cache_data
def display_groupwise_scatter_row(ests, blind_mode_df, statistic):
    # Calculate group statistics
    group_stats = calculate_group_statistics(ests, blind_mode_df, statistic=statistic)
    # Create charts
    equality_line = create_equality_line()
    chart_fe = create_scatter_plot(
        group_stats, "All_Participants", "Forcasting_experience"
    )
    chart_lw = create_scatter_plot(group_stats, "All_Participants", "LessWrong_member")
    chart_sf = create_scatter_plot(group_stats, "All_Participants", "Super forecaster")
    # Display charts using Streamlit
    scatter_column_median = st.columns(3)
    with scatter_column_median[0]:
        st.altair_chart(equality_line + chart_fe, use_container_width=True)
    with scatter_column_median[1]:
        st.altair_chart(equality_line + chart_lw, use_container_width=True)
    with scatter_column_median[2]:
        st.altair_chart(equality_line + chart_sf, use_container_width=True)
    return


@st.cache_data
def run_ne_calculations(flattened_prediction_df, ne):
    """Run the calculations for the not equal to ne analysis."""
    fractions = util.calculate_not_equal_prediction_fractions(
        flattened_prediction_df, ne
    )
    # Perform chi-square tests and store results
    chi_square_results = []
    for cls in ["FE", "SF", "LW"]:
        stat, p = util.perform_ne_chi_square_test(
            fractions["all"],
            len(flattened_prediction_df[flattened_prediction_df["class"] == "all"]),
            fractions[cls],
            len(flattened_prediction_df[flattened_prediction_df["class"] == cls]),
        )
        chi_square_results.append({"class": cls, "stat": stat, "p": p})
    chi_square_results.append({"class": "all", "stat": np.nan, "p": np.nan})
    chi_square_results = pd.DataFrame(chi_square_results)
    chi_square_results["percent"] = [
        fractions[cls] * 100 for cls in flattened_prediction_df["class"].unique()
    ]
    # chi_square_results.sort_values(by="percent", ascending=False, inplace=True)
    chi_square_results.sort_values(by="class", ascending=False, inplace=True)
    # Perform bootstrap
    # Use fewer iterations for GitHub Actions or pytest
    if os.getenv("GITHUB_ACTIONS") == "true" or "pytest" in sys.modules:
        n_iterations = 5
    else:
        n_iterations = 1000
    fractions_dist = util.perform_frac_ne_bootstrap(
        flattened_prediction_df, ne, n_iterations=n_iterations, silent=True
    )
    return chi_square_results, fractions_dist


@st.cache_data
def get_group_n(data):
    """Get the number of participants in each group.

    Parameters:
    - data: DataFrame containing additional segmentation data.

    Returns:
    - group_n: a DataFrame of group sizes for
        "All participants", "Forcasting experience" (FE),
        "LessWrong member" (LW), and "Super forecaster" (SF)
    """
    group_n = pd.DataFrame()
    group_n["All_Participants"] = [len(data)]
    group_n["Forcasting_experience"] = [
        len(data.loc[data["ForecastingExperience"].values == "Yes"])
    ]
    group_n["LessWrong_member"] = [len(data.loc[data["LessWrong"].values == "Yes"])]
    group_n["Super forecaster"] = [
        len(data.loc[data["Superforecaster"].values == "Yes"])
    ]
    return group_n


@st.cache_data
def create_segmented_prediction_histogram(
    data, selected_question_number, n_bins=10, bin_range=(0, 100)
):
    """
    Creates overlapping prediction histograms for a given question, split by group.

    Parameters:
    - data: DataFrame containing additional segmentation data.
    - selected_question_number: Question number.
    - n_bins: Number of bins for the histogram.
    - bin_range: Range of bins.

    Returns:
    - Altair chart object.
    """
    # Match the full column name from the question string
    question_substring = "@" + str(selected_question_number) + "."
    matching_column = [col for col in data.columns if question_substring in col][0]
    if not matching_column:
        raise ValueError(
            f"No columns found containing the string '{question_substring}'"
        )
    # TODO: Convert these to a function
    # Compute histogram data for all
    tmp = data[matching_column].squeeze().dropna()
    hist, bin_edges = np.histogram(tmp, bins=n_bins, range=bin_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_norm = hist / hist.sum()
    # Compute histogram data for ForecastingExperience
    ests_FE = data.loc[data["ForecastingExperience"].values == "Yes"]
    tmp_FE = ests_FE.filter(like=question_substring, axis=1).squeeze().dropna()
    hist, bin_edges = np.histogram(tmp_FE, bins=n_bins, range=bin_range)
    bin_centers_FE = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_norm_FE = hist / hist.sum()
    # Compute histogram data for LessWrong
    ests_LW = data.loc[data["LessWrong"].values == "Yes"]
    tmp_LW = ests_LW.filter(like=question_substring, axis=1).squeeze().dropna()
    hist, bin_edges = np.histogram(tmp_LW, bins=n_bins, range=bin_range)
    bin_centers_LW = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_norm_LW = hist / hist.sum()
    # Compute histogram data for superforecasters
    ests_SF = data.loc[data["Superforecaster"].values == "Yes"]
    tmp_SF = ests_SF.filter(like=question_substring, axis=1).squeeze().dropna()
    hist, bin_edges = np.histogram(tmp_SF, bins=n_bins, range=bin_range)
    bin_centers_SF = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist_norm_SF = hist / hist.sum()
    # Combine data for plotting
    source = pd.DataFrame(
        {
            "Experience": np.concatenate(
                (
                    np.repeat("All participants", len(hist_norm)),
                    np.repeat("Forcasting experience", len(hist_norm_FE)),
                    np.repeat("LessWrong member", len(hist_norm_LW)),
                    np.repeat("Super forecaster", len(hist_norm_SF)),
                ),
            ),
            "x": np.concatenate(
                (bin_centers, bin_centers_FE, bin_centers_LW, bin_centers_SF)
            ),
            "y": np.concatenate((hist_norm, hist_norm_FE, hist_norm_LW, hist_norm_SF)),
        }
    )
    # Create the Altair chart
    chart = (
        alt.Chart(source, height=450)
        .mark_line(opacity=1.0, point=True)
        .encode(
            x=alt.X(
                "x",
                axis=alt.Axis(values=bin_edges, grid=False),
                bin=alt.Bin(maxbins=n_bins, extent=bin_range),
                title="Predicted probability (binned)",
            ),
            y=alt.Y(
                "y", axis=alt.Axis(format="%"), title="% of predictions", stack=False
            ),
            color=alt.Color(
                "Experience",
                scale=alt.Scale(
                    range=["white", "yellow", "orange", "red"]
                ),  # ["#1FC3AA", "#8624F5"]
                legend=alt.Legend(direction="horizontal", orient="top"),
            ),
        )
    )
    return chart


@st.cache_data
def calculate_group_statistics(ests, data, statistic="median"):
    """
    Calculates group statistics (mean or median) for each predictor group.

    Parameters:
    - ests: DataFrame containing the estimates.
    - data: DataFrame containing additional segmentation data.
    - statistic: 'mean' or 'median' to specify the statistic to be calculated.

    Returns:
    - DataFrame with calculated statistics for each group.
    """
    if statistic not in ["mean", "median"]:
        raise ValueError("Statistic must be either 'mean' or 'median'")

    stat_function = getattr(pd.DataFrame, statistic)
    group_stats = pd.DataFrame()
    group_stats["All_Participants"] = stat_function(ests, axis=0).values
    group_stats["Forcasting_experience"] = stat_function(
        ests.loc[data["ForecastingExperience"].values == "Yes"], axis=0
    ).values
    group_stats["LessWrong_member"] = stat_function(
        ests.loc[data["LessWrong"].values == "Yes"], axis=0
    ).values
    group_stats["Super forecaster"] = stat_function(
        ests.loc[data["Superforecaster"].values == "Yes"], axis=0
    ).values
    return group_stats


@st.cache_data
def linear_regression_stats(x, y):
    """
    Performs linear regression and calculates the slope, angle, and p-value.

    Parameters:
    - x: x-values for the regression.
    - y: y-values for the regression.

    Returns:
    - slope: Slope of the regression line.
    - angle: Angle of the slope in degrees.
    - p_value: p-value of the regression.
    """
    slope, _, _, p_value, _ = stats.linregress(x, y)
    angle = np.rad2deg(np.arctan(slope))
    return slope, angle, p_value


@st.cache_data
def create_scatter_plot(median_pred, x_col, y_col, fit_line=True):
    """
    Creates a scatter plot comparing two sets of group statistics.

    Parameters:
    - median_pred: DataFrame with group statistics.
    - x_col: Column name for x-axis.
    - y_col: Column name for y-axis.
    - fit_line: Boolean indicating whether to fit a regression line.

    Returns:
    - Altair chart object.
    """
    chart = (
        alt.Chart(median_pred, height=250)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                x_col, scale=alt.Scale(domain=(0, 100)), title=x_col.replace("_", " ")
            ),
            y=alt.Y(
                y_col, scale=alt.Scale(domain=(0, 100)), title=y_col.replace("_", " ")
            ),
            tooltip=[
                "All_Participants",
                "Forcasting_experience",
                "LessWrong_member",
                "Super forecaster",
            ],
        )
    )
    # Add linear regression line and statistics annotation
    if fit_line:
        # Calculate linear regression
        # Note: p_value is for the deviation from the equality line
        _, angle, _ = linear_regression_stats(median_pred[x_col], median_pred[y_col])
        _, _, p_value = linear_regression_stats(
            median_pred[x_col], median_pred[y_col] - median_pred[x_col]
        )
        # Create regression line
        regression_line = chart.transform_regression(
            x_col, y_col, extent=[0, 100]
        ).mark_line(color="red", opacity=0.5, clip=True)
        # Regression stats annotation to add to chart
        ticker_x = 4
        ticker_y = 94
        regresssion_pvalue = p_value
        regression_slope = angle
        ticker = (
            f"p = {regresssion_pvalue:.3f}",
            f"∠ = {regression_slope:.1f}°",
        )
        annotations_df = pd.DataFrame(
            [(ticker_x, ticker_y, regresssion_pvalue)], columns=["x", "y", "p"]
        )
        annotation_layer = (
            alt.Chart(annotations_df)
            .mark_text(size=15, text=ticker, align="left", color="red")
            .encode(
                x=alt.Y("x:Q"),
                y=alt.Y("y:Q"),
                tooltip=["p"],
            )
        )
        chart_final = alt.layer(chart, annotation_layer, regression_line)
    return chart_final


@st.cache_data
def create_equality_line():
    """Equality line for reference"""
    return (
        alt.Chart(pd.DataFrame({"x": [0, 100], "y": [0, 100]}))
        .mark_line(strokeDash=[5, 2.5], color="grey", opacity=0.5)
        .encode(x="x", y="y")
    )


def main():
    # Configure page
    Home.configure_page(page_title="Predictions by experience")

    # Load data
    blind_mode_df, markets_df, resolution_vector = Home.load_data()
    ests = blind_mode_df.filter(like="@", axis=1)
    size_of_groups = get_group_n(blind_mode_df)

    # Introduction to page
    st.title("Does self-reported forecasting experience correlate with predictions?")
    st.markdown(
        f"""
        Blind-mode participants self-identified among several categories of forecasting 
        experience.
        There were {size_of_groups["All_Participants"][0]} total participaints.
        {size_of_groups["Forcasting_experience"][0]} participants reported having
        forecasting experience, 
        {size_of_groups["LessWrong_member"][0]} participants reported being members of
        LessWrong, and
        {size_of_groups["Super forecaster"][0]} participants reported being super
        forecasters.
        """
    )

    # Slider to select question
    st.divider()
    st.subheader("Prediction distributions by group")
    selected_question_number = st.slider("Select question", 1, 50, 1)

    # Write question text and outcome
    full_question = Home.get_question_text(markets_df, selected_question_number)
    question_outcome_string = Home.get_question_outcome_string(
        resolution_vector, selected_question_number
    )
    question_outcome_color = "green" if question_outcome_string == "Yes" else "red"
    st.markdown(
        f""" 
        Question: {full_question} <br>
        Outcome: :{question_outcome_color}[{question_outcome_string}]
        """,
        unsafe_allow_html=True,
    )

    # Prediction histogram split by predictor group
    segmented_chart = create_segmented_prediction_histogram(
        blind_mode_df, selected_question_number
    )
    st.altair_chart(segmented_chart, use_container_width=True)

    # Group statistics for the histogram (can be mean or median)
    st.markdown("""Question mean by group""")
    display_group_metrics(ests, blind_mode_df, selected_question_number)
    st.divider()

    # Compare average predictions for each question, split by predictor group
    st.subheader("Average predictions are similar across self-identified groups")
    # Median
    st.markdown(
        "Median predictions for each question split by group: all vs {FE, LW, SF}"
    )
    display_groupwise_scatter_row(ests, blind_mode_df, statistic="median")
    # Mean
    st.markdown(
        "Mean predictions for each question split by group: all vs {FE, LW, SF}"
    )
    display_groupwise_scatter_row(ests, blind_mode_df, statistic="mean")
    st.markdown(
        """It turns out that the average predictions are similar across groups,
        whether you look at the mean or the median, and whether you compare those \
        with forecasting experience, LessWrong members, or superforecasters.\
        However, there are some intersting differences in the prediction distributions.
    """
    )
    st.divider()

    # TODO: Move below into a function?

    # Comparison from combining all questions
    # Flatten the prediction df to aggregate all predictions by group
    flattened_prediction_df = process.flatten_prediction_df(blind_mode_df)
    rounded_values = list(range(5, 100, 5))
    extreme_values = [1, 99]
    ne_options = [
        extreme_values,
        rounded_values,
        extreme_values + rounded_values,
    ]
    # Perform calculations comparing fractions to ne
    # run_ne_calculations runs chi-square test and returns bootstrap distributions
    ne_results_df = {}
    for i, ne in enumerate(ne_options):
        chi_square_results, fractions_dist = run_ne_calculations(
            flattened_prediction_df, ne
        )
        ne_results_df[i] = (chi_square_results, fractions_dist)

    # ne = [1, 99]
    st.subheader(
        """The LessWrong and Forecasting Experience groups are more likely to\
            avoid predicting 1% or 99%, but not the Superforecasters group"""
    )
    st.markdown("Percent of all predictions that are not equal to 1 or 99.")
    i_ne = 0
    chi_square_results, fractions_dist = ne_results_df[i_ne]
    st.markdown("#### Chi-square test results comparing each group to all")
    st.dataframe(chi_square_results.set_index("class"))
    st.markdown("#### Bootstrap distributions")
    fig = create_frac_ne_bootstrap_distributions(fractions_dist, n_bins=None)
    fig.encoding.x.title = "Fraction of predictions that are not equal to 1 or 99"
    st.altair_chart(fig, use_container_width=True)
    st.divider()

    # ne = list(range(5, 100, 5))
    st.subheader(
        """Those with forecasting experience are less likely to round their predictions\
        by 5s.
        """
    )
    st.markdown("Percent of all predictions that are not rounded to the nearest 5.")
    i_ne = 1
    chi_square_results, fractions_dist = ne_results_df[i_ne]
    st.markdown("#### Chi-square test results comparing each group to all")
    st.dataframe(chi_square_results.set_index("class"))
    st.markdown("#### Bootstrap distributions")
    fig = create_frac_ne_bootstrap_distributions(fractions_dist, n_bins=None)
    fig.encoding.x.title = (
        "Fraction of predictions that are not rounded to the nearest 5"
    )
    st.altair_chart(fig, use_container_width=True)
    st.divider()

    # ne = [1, 99] + list(range(5, 100, 5))
    st.subheader(
        """All groups are more likely to avoid both rounding and extremes.
        """
    )
    st.markdown("Percent of all predictions that are neither rounded nor extreme.")
    i_ne = 2
    chi_square_results, fractions_dist = ne_results_df[i_ne]
    st.markdown("#### Chi-square test results comparing each group to all")
    st.dataframe(chi_square_results.set_index("class"))
    st.markdown("#### Bootstrap distributions")
    fig = create_frac_ne_bootstrap_distributions(fractions_dist, n_bins=None)
    fig.encoding.x.title = (
        "Fraction of predictions that are neither rounded nor extreme"
    )
    st.altair_chart(fig, use_container_width=True)

    return


if __name__ == "__main__":
    main()
