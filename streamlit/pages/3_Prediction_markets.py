# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

from src import process
from src import util
import Home


# Functions for this page
@st.cache_data
def plot_my_score_time_series(my_pred_df, market_hist_df):
    mse_chart, brier_chart = create_my_score_time_series(my_pred_df, market_hist_df)
    st.altair_chart(mse_chart, use_container_width=True)
    st.altair_chart(brier_chart, use_container_width=True)
    return None


@st.cache_data
def plot_bier_violin(blind_mode_final_brier, blind_mode_df):
    fig = create_brier_violin(blind_mode_final_brier, blind_mode_df)
    st.plotly_chart(fig)
    return None


@st.cache_data
def create_my_score_time_series(my_pred_df, market_hist_df, transient=0, height=250):
    """Calculates and returns Altair charts for the MSE and Brier score time series for all questions."""
    my_mse = process.get_score_df(my_pred_df, market_hist_df, score_type="mse")
    my_brier = process.get_score_df(my_pred_df, market_hist_df, score_type="brier")
    mse_df = pd.DataFrame(
        {
            "Time": [
                datetime.datetime.fromtimestamp(time / 1000) for time in my_mse["time"]
            ][transient:],
            "MSE": my_mse[transient:][0],
        }
    )
    brier_df = pd.DataFrame(
        {
            "Time": [
                datetime.datetime.fromtimestamp(time / 1000)
                for time in my_brier["time"]
            ][transient:],
            "Brier Score": my_brier[transient:][0],
        }
    )
    mse_chart = (
        alt.Chart(mse_df, height=height)
        .mark_line()
        .encode(
            x=alt.X("Time", axis=alt.Axis(format="%Y-%m")),
            y="MSE",
            tooltip=["Time", "MSE"],
        )
    )
    brier_chart = (
        alt.Chart(brier_df, height=height)
        .mark_line()
        .encode(
            x=alt.X("Time", axis=alt.Axis(format="%Y-%m")),
            y="Brier Score",
            tooltip=["Time", "Brier Score"],
        )
    )
    return mse_chart, brier_chart


@st.cache_data(show_spinner=False)
def plot_question_market_time_series(
    selected_question_number,
    selected_question_prediction,
    market_hist_df,
    plot_market_subset=True,
):
    fig_1, fig_2, fig_3 = create_question_market_time_series(
        selected_question_number,
        selected_question_prediction / 100,
        market_hist_df,
        plot_subset=plot_market_subset,
    )
    st.altair_chart(fig_1, use_container_width=True)
    st.altair_chart(fig_2, use_container_width=True)
    st.altair_chart(fig_3, use_container_width=True)
    return None


@st.cache_data
def create_question_market_time_series(
    q_to_plot,
    q_to_plot_prediction,
    market_hist_df,
    plot_subset=True,
    chart_height=150,
    hide_initial_x_axis=True,
):
    """Calculates and returns Altair charts for Market history, RMSE, and
    Brier score for a question.
    Note: calculation block runs in ~0.01 s, but page takes much longer to run.
        Not sure why.
    """
    q_to_plot = f"Q{q_to_plot}"
    time_vec = pd.to_datetime(market_hist_df["time"], unit="ms")
    question_most_likely_outcome = np.array(market_hist_df[q_to_plot]).round()
    rmse_vec = np.sqrt(np.square(q_to_plot_prediction - market_hist_df[q_to_plot]))
    brier_score_vec = np.sqrt(
        np.square(q_to_plot_prediction - question_most_likely_outcome)
    )
    source = pd.DataFrame(
        {
            "Time": time_vec,
            "Probability": market_hist_df[q_to_plot],
            "RMSE": rmse_vec,
            "Brier Score": brier_score_vec,
        }
    )
    if plot_subset:
        # Plot only every 10th point
        source = source.iloc[::10, :]
    # Create Altair charts
    chart_1 = (
        alt.Chart(source, height=chart_height)
        .mark_line()
        .encode(
            x=alt.X("Time", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y(
                "Probability",
                scale=alt.Scale(domain=[0, 1]),
                title="Market Probability",
            ),
            tooltip=["Time", "Probability"],
        )
    )
    chart_2 = (
        alt.Chart(source, height=chart_height)
        .mark_line()
        .encode(
            x=alt.X("Time", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("RMSE", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Time", "RMSE"],
        )
    )
    chart_3 = (
        alt.Chart(source, height=chart_height)
        .mark_line()
        .encode(
            x=alt.X("Time", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("Brier Score", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Time", "Brier Score"],
        )
    )
    if hide_initial_x_axis:
        chart_1 = chart_1.encode(
            x=alt.X(
                "Time",
                axis=alt.Axis(ticks=False, labels=False, domain=False, title=None),
            ),
        )
        chart_2 = chart_2.encode(
            x=alt.X(
                "Time",
                axis=alt.Axis(ticks=False, labels=False, domain=False, title=None),
            ),
        )
        # Increase height of chart_3 to compensate for hidden x-axis
        chart_3 = chart_3.properties(height=chart_height + 75)
    return chart_1, chart_2, chart_3


@st.cache_data
def create_brier_violin(blind_mode_final_brier, blind_mode_df):
    briers_all = blind_mode_final_brier
    briers_fe = blind_mode_final_brier[blind_mode_df["ForecastingExperience"] == "Yes"]
    briers_sf = blind_mode_final_brier[blind_mode_df["Superforecaster"] == "Yes"]
    briers_lw = blind_mode_final_brier[blind_mode_df["LessWrong"] == "Yes"]
    source = pd.DataFrame(
        {
            "Brier score": np.concatenate(
                (briers_all, briers_fe, briers_lw, briers_sf)
            ),
            "Type": np.concatenate(
                (
                    ["All"] * len(briers_all),
                    ["Forecasting experience"] * len(briers_fe),
                    ["LessWrong"] * len(briers_lw),
                    ["Superforecaster"] * len(briers_sf),
                )
            ),
        }
    )
    # Match colors of set1 from https://vega.github.io/vega/docs/schemes/
    # [white, yellow, orange, red]
    colors = ["#FFFFFF", "#ffff33", "#ff7f00", "#e41a1c"]
    fig = (
        px.violin(
            source,
            y="Brier score",
            x="Type",
            box=True,
            points="all",
            color="Type",
            color_discrete_sequence=colors,
            hover_data=source.columns,
        )
        .update_layout(showlegend=False)
        .update_traces(marker_size=2)
        .update_xaxes(title=None)
    )
    return fig


@st.cache_data
def get_market_df():
    market_hist_df = process.get_all_markets(silent=True)
    return market_hist_df


def main():
    # Configure page
    Home.configure_page(page_title="Prediction markets")

    # Load data
    blind_mode_df, markets_df, resolution_vector = Home.load_data()
    my_predictions, _ = Home.load_predictions()
    market_hist_df = get_market_df()
    estimates_matrix = Home.get_estimates_matrix(blind_mode_df)
    blind_mode_final_brier = np.mean(
        np.square(estimates_matrix - resolution_vector["resolution"].values), axis=1
    )

    # Introduction to page
    st.write(
        """# How well do our aggregate predictions compare to prediction\
                markets across the year?"""
    )
    st.markdown(
        """
        [Manifold Markets](<https://manifold.markets/>)
        has a prediction
        [market](<https://manifold.markets/ACXBot?s=newest&f=all&tab=questions>) 
        for each question in the contest.
        We can use the Manifold Markets API to get the time series of the forecast
        score for each question.
        We can then evaluate our predications against the market across time.
        As the markets get closer to their close (January 1st, 2024 at the latest),
        they should converge to the true outcome.
        At each point in time, we can evaluate our predictions against the market and
        see how we are doing.

        As before, we will use the Brier score to evaluate our predictions.
        But since the Brier score is only defined for binary outcomes,
        we will use at each point in time the most likely outcome of the market
        when calculating the Brier score.

        We will also use the root mean squared error (RMSE) to evaluate our predictions.
        The Brier score, as we are calculating it,
        only changes when the market passes the 50% threshold.
        The RMSE, on the other hand, smoothly changes with the market's predictions.

        """
    )
    st.divider()

    # Show market history, RMSE, and Brier score for selectable question and prediction
    st.subheader("Market time series")
    st.markdown(
        """
        Select a question and a prediction to see the time series of the market.
        For both the Brier score and the RMSE
        a lower value indicates a more accurate prediction.
        """
    )
    selected_question_number = st.slider("Select question", 1, 50, 46)
    initial_prediction_value = int(my_predictions[(selected_question_number) - 1] * 100)
    selected_question_prediction = st.slider(
        "Make a prediction (defaults to our aggregate prediction)",
        1,
        99,
        initial_prediction_value,
    )
    full_question = Home.get_question_text(markets_df, selected_question_number)
    question_outcome_string = Home.get_question_outcome_string(
        resolution_vector, selected_question_number
    )
    question_outcome_color = "green" if question_outcome_string == "Yes" else "red"
    st.markdown(
        f""" 
        Question: {full_question} <br>
        Actual outcome: :{question_outcome_color}[{question_outcome_string}]
        """,
        unsafe_allow_html=True,
    )
    plot_market_subset = not st.checkbox("Plot all time points (slow)", value=False)
    plot_question_market_time_series(
        selected_question_number,
        selected_question_prediction,
        market_hist_df,
        plot_market_subset=plot_market_subset,
    )
    st.divider()
    
    # Evaluate time series of the aggregated predictions
    st.subheader("Aggregated predictions")
    st.markdown(
        """
        How did our aggregated predictions do over time?
        Our aggregated predictions start off strong (lower is better),
        meaning they are in accordance with the markets' predictions.
        The scores increase over time, as the market incorporates new
        information that wasn't available to us or the the Blind Mode participants
        at the start of the contest.
        """
    )
    my_pred_df = pd.DataFrame(columns=["Q" + str(i) for i in range(1, 51)])
    my_pred_df.loc[0] = my_predictions
    plot_my_score_time_series(my_pred_df, market_hist_df)
    st.divider()

    # Evalute final Brier scores of all Blind Mode participants
    st.subheader("Final scores")
    st.markdown(
        """
        How did our aggregated predictions do against Blind Mode participants
        at the end of the contest?
        """
    )
    plot_bier_violin(blind_mode_final_brier, blind_mode_df)
    aggregated_final_brier = np.mean(
        np.square(my_pred_df.values - resolution_vector["resolution"].values), axis=1
    )
    blind_mode_final_brier.sort()
    my_rank = np.searchsorted(blind_mode_final_brier, aggregated_final_brier)
    my_percentile = my_rank / len(blind_mode_final_brier)
    st.markdown(
        f"""
        Our aggregate predictions had a final Brier score of 
        {aggregated_final_brier[0]:.3f}.
        The top 10 Blind Mode participants had final Brier scores of 
        ${blind_mode_final_brier[:10].round(3).tolist()}$.
        The aggregate score would have placed {util.ordinal(my_rank[0])}
        which is the top {my_percentile[0]:.2%}.
        As predicted by our simulation analysis, even with perfectly calibrated
        predictions, we would still be unlikely to win in a field of
        {len(blind_mode_final_brier)} participants.
        """
    )
    st.divider()

    # Link to notebook
    st.markdown(
        """
        To see detailed time series analysis of the blind mode participants see
        the notebook 
        [`./notebooks/3_manifold.ipynb`](<https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/3_manifold.ipynb>)
        of this project's GitHub repo.
        """
    )

    return


if __name__ == "__main__":
    main()
