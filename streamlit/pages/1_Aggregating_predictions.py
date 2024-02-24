# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import beta
import streamlit as st

import Home


# Functions for page
@st.cache_data
def create_equality_line():
    return (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[15, 10], color="black", opacity=1.0, strokeWidth=4.0)
        .encode(x="x", y="y")
    )


@st.cache_data
def create_my_preds_df(my_preds, resolution_vector, markets_df):
    all_question_text = [
        Home.get_question_text(markets_df, x)
        for x in resolution_vector["question_number"]
    ]
    all_question_outcomes = [
        "Yes" if x else "No" for x in resolution_vector["resolution"]
    ]
    my_preds_df = pd.DataFrame(
        my_preds.values.T, columns=resolution_vector["question_number"]
    )
    my_preds_df = my_preds_df.T
    my_preds_df = my_preds_df.rename(columns={0: "Prediction"})
    my_preds_df["Outcome"] = (np.array(all_question_outcomes)).T
    my_preds_df["Question text"] = [
        x.split(". ")[1] for x in np.array(all_question_text)
    ]
    my_preds_df.index.name = "Question"
    return my_preds_df


@st.cache_data
def plot_weighted_beta_scatter(blind_mode_df):
    fig, my_preds = create_weighted_beta_scatter_plot(blind_mode_df)
    scatter_column_2 = st.columns([1, 2, 1])
    with scatter_column_2[1]:
        st.altair_chart(fig, use_container_width=True)
    return my_preds


@st.cache_data
def plot_beta_scatter(blind_mode_df):
    fig = create_beta_scatter_plot(blind_mode_df)
    scatter_column_1 = st.columns([1, 2, 1])
    with scatter_column_1[1]:
        st.altair_chart(fig, use_container_width=True)
    return None


@st.cache_data
def create_beta_scatter_plot(blind_mode_df, beta_a=1 / 7, beta_b=1 / 7):
    ests = blind_mode_df.filter(like="@", axis=1)
    original_predictions = ests.mean() / 100
    transformed_predictions = beta.ppf(original_predictions, beta_a, beta_b)
    aggregate_predictions_df = pd.DataFrame(
        {
            "Original predictions": original_predictions,
            "Transformed predictions": transformed_predictions,
        }
    )
    chart = (
        alt.Chart(aggregate_predictions_df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "Original predictions",
                title="Mean prediction",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "Transformed predictions",
                title="Beta transformed values",
                scale=alt.Scale(domain=[0, 1]),
            ),
        )
    )
    ref_line = create_equality_line()
    return chart + ref_line


@st.cache_data
def create_weighted_beta_scatter_plot(
    blind_mode_df, weights=None, beta_a=1 / 3, beta_b=1 / 3
):
    if weights is None:
        weights = [0.05, 0.8, 0.1, 0.05]
    ests = blind_mode_df.filter(like="@", axis=1)
    # Weight the estimates
    ests_none = ests
    ests_FE = ests.loc[blind_mode_df["ForecastingExperience"].values == "Yes"]
    ests_SF = ests.loc[blind_mode_df["Superforecaster"].values == "Yes"]
    ests_LW = ests.loc[blind_mode_df["LessWrong"].values == "Yes"]
    weighted_sum = (
        weights[0] * ests_none.mean(axis=0)
        + weights[1] * ests_SF.mean(axis=0)
        + weights[2] * ests_FE.mean(axis=0)
        + weights[3] * ests_LW.mean(axis=0)
    )
    # Apply the beta transformation to the weighted mean predictions
    weighted_tranformed_mean = beta.ppf(weighted_sum / 100, beta_a, beta_b)
    my_preds = pd.DataFrame(np.round(weighted_tranformed_mean * 100))
    my_preds[my_preds > 99] = 99
    my_preds[my_preds < 1] = 1
    aggregate_predictions_df = pd.DataFrame()
    aggregate_predictions_df["Original predictions"] = ests.mean() / 100
    aggregate_predictions_df["Weighted transformed prediction"] = my_preds.values / 100
    aggregate_predictions_df["Superforecaster mean"] = ests_SF.mean() / 100
    chart = (
        alt.Chart(aggregate_predictions_df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "Superforecaster mean",
                title="Superforecaster mean",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "Weighted transformed prediction",
                title="Weighted transformed prediction",
                scale=alt.Scale(domain=[0, 1]),
            ),
        )
    )
    ref_line = create_equality_line()
    return chart + ref_line, my_preds


def main():
    # Configure page
    Home.configure_page(page_title="Aggregating predictions")

    # Load data
    blind_mode_df, markets_df, resolution_vector = Home.load_data()

    # Introduction to page
    st.write(
        """# Can we aggregate Blind Mode predictions\
            to generate more accurate predictions?"""
    )
    st.markdown(
        """
        Aggregations of predictions are often more accurate than individual predictions.
        Since we have the predictions of many participants in the Blind Mode, we can
        use this data to generate our own aggregated prediction.

        There are several approaches to aggregation.
        The most straightforward is to take the average of the predictions, either the 
        mean or median. 
        However, this approach has two main deficiencies that we can easily address.
        First, taking the mean or median has a centralizing tendency which empirically
        leads to underconfidence.
        Second, a simple aggregation does not account for the fact that some
        participants are better predictors than others.
        """
    )
    st.divider()

    # First aggregation method
    st.subheader("Beta transformed arithmetic mean")
    st.markdown(
        r"""[Hanea et al., 2021](https://doi.org/10.1371/journal.pone.0256919) analyzed
        previously collected forecasting data and evaluated several methods for
        aggregating predictions.
        They found that the beta transformed arithmetic mean (BetaArMean) outperformed
        other aggregation methods on all of their data sets.
        The BetaArMean involves caculating the mean prediction and then transforming it
        using the cumulative distribution function of a beta distribution,
        effectively extremising the aggregate.
        The beta distribution is a continuous probability distribution with two
        parameters, $\alpha$ and $\beta$.
        They found that $\alpha=\beta=7$ performed the best the aggregating predictions
        in their data sets.
        """
    )

    # Beta transformation with alpha=beta=7
    plot_beta_scatter(blind_mode_df)
    st.divider()

    # Second aggregation method
    st.subheader("Beta transformed experience-weighted arithmetic mean")
    st.markdown(
        """
        Although the mean predictions didn't systematically differ to a large extend
        across groups of participants, there were differences in individual predictions.
        So 
        I shook my magic 8-ball and decided to use the weights of 
        $[0.05, 0.8, 0.1, 0.05]$ for $[All, SF, FE, LW]$ participants.
        I also thought the extremization was a bit too much,
        so I toned it down.

        Those tweaks resulted in the following predictions,
        comparing the final aggreagated predictions (y-axis) to the mean predictions
        of the Superforecasters (x-axis).
        """
    )
    my_preds = plot_weighted_beta_scatter(blind_mode_df)

    # Show final predictions
    st.markdown(
        """Now we just need to clean up a bit.
        Predictions must be integer values between $1$ and $99$,
        which leaves us with the following predictions:
        """
    )
    my_preds_df = create_my_preds_df(my_preds, resolution_vector, markets_df)
    st.dataframe(my_preds_df, height=500)

    return


if __name__ == "__main__":
    main()
