# pylint: disable=invalid-name
"""The home page for the ACX 2023 Prediction Contest streamlit app.

Hosted at https://acx-predictions.streamlit.app/

Plotting approach for streamlit is to do as much necessary data transformation as 
possible prior to calling altair to create a plot.
e.g. use st.mark_bar to plot pre-calculated histogram data.
"""

## Imports
import sys
import os

import altair as alt
import numpy as np
import pandas as pd
import pyprojroot.here
import streamlit as st
from streamlit_extras.card import card
from streamlit_extras.switch_page_button import switch_page
import matplotlib.pyplot as plt

sys.path.insert(0, str(pyprojroot.here()))  # Add parent directory to path
from src import process  # pylint: disable=wrong-import-position


# Constants used across pages
APP_ICON_PATH = os.path.join(pyprojroot.here(), "assets/images/favicons/icon.png")
GH_ICON_PATH = os.path.join(
    pyprojroot.here(), "assets/images/favicons/github-mark-white.svg"
)


# Functions used across pages
@st.cache_data
def get_estimates_matrix(df, nan_method="median"):
    return process.get_estimates_matrix(df, nan_method=nan_method)


@st.cache_data
def load_predictions():
    """Load previously saved prediction files"""
    # Mean superforecaster predictions
    data_file = os.path.join(process.RESULTS_FOLDER, "sf_mean_predictions.csv")
    sf_mean_predictions_data = pd.read_csv(data_file)
    sf_mean_predictions = (sf_mean_predictions_data.values.T[0] / 100).round(2)
    # My submitted predictions
    data_file = os.path.join(process.RESULTS_FOLDER, "my_final_predictions.csv")
    my_predictions_data = pd.read_csv(data_file)
    my_predictions = my_predictions_data.values.T[0] / 100
    return my_predictions, sf_mean_predictions


def add_sidebar_links():
    with st.sidebar:
        logo_column_1 = st.columns([0.6, 3.4])
        with logo_column_1[0]:
            st.image(APP_ICON_PATH)
        with logo_column_1[1]:
            st.link_button(
                " jbreffle.github.io   ",
                "https://jbreffle.github.io/",
                type="primary",
                use_container_width=True,
            )
        logo_column_2 = st.columns([0.6, 3.4])
        with logo_column_2[0]:
            st.image(GH_ICON_PATH)
        with logo_column_2[1]:
            st.link_button(
                "github.com/jbreffle",
                "https://github.com/jbreffle",
                type="primary",
                use_container_width=True,
            )
    return


def expand_sidebar_pages_vertical_expander():
    st.markdown(
        """
    <style>
    div[data-testid='stSidebarNav'] ul {max-height:none}</style>
    """,
        unsafe_allow_html=True,
    )
    return


def hide_imgae_fullscreen():
    """Hides the fullscreen button for images."""
    hide_img_fs = """
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        """
    st.markdown(hide_img_fs, unsafe_allow_html=True)
    return


@st.cache_data
def get_question_outcome_string(resolution_vector, selected_question_number):
    question_outcome = resolution_vector["resolution"][selected_question_number - 1]
    question_outcome_string = (
        "Yes" if question_outcome == 1 else "No" if question_outcome == 0 else "N/A"
    )
    return question_outcome_string


@st.cache_data
def get_question_text(markets_df, selected_question_number):
    output = markets_df.loc[
        markets_df["question_number"] == int(selected_question_number)
    ]["question"].values[0]
    return output


@st.cache_data
def set_plt_style():
    params = {
        "ytick.color": "w",
        "xtick.color": "w",
        "axes.labelcolor": "w",
        "axes.edgecolor": "w",
        "text.color": "w",
        "grid.color": "w",
        "grid.linewidth": 0.1,
        "axes.linewidth": 0.25,
        "axes.grid": True,
        "font.size": 8,
        "axes.titlepad": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 12,
    }
    plt.rcParams.update(params)
    return


def configure_page(page_title=None):
    """Convenience function to configure the page layout."""
    st.set_page_config(
        page_title=page_title,
        page_icon=APP_ICON_PATH,
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    add_sidebar_links()
    hide_imgae_fullscreen()
    expand_sidebar_pages_vertical_expander()
    set_plt_style()
    return


@st.cache_data
def load_data():
    if "blind_mode_df" in st.session_state:
        return (
            st.session_state.blind_mode_df,
            st.session_state.markets_df,
            st.session_state.resolution_vector,
        )
    else:
        data_loaded, _ = process.load_and_process_results()
        st.session_state.blind_mode_df = data_loaded
        markets_df = process.get_current_probs(silent=True)
        st.session_state.markets_df = markets_df
        resolution_vector = process.get_target_df()
        st.session_state.resolution_vector = resolution_vector
        return data_loaded, markets_df, resolution_vector


# Page-specific functions
@st.cache_data
def create_prediction_histogram(
    data, selected_question_number, n_bins=10, bin_range=(0, 100)
):
    """
    Creates a histogram of predictions for a given question.

    Note that question 9 has the tallest peak at ~0.09 density.
    Based on src.plot.demo_pred_hist.

    Parameters:
    - data: DataFrame containing the estimates.
    - selected_question_number: Question number.
    - n_bins: Number of bins for the histogram.
    - bin_range: Range of bins.

    Returns:
    - Altair chart object.
    """
    # Match the full column name from the question number
    question_substring = "@" + str(selected_question_number) + "."
    matching_column = [col for col in data.columns if question_substring in col][0]
    if not matching_column:
        raise ValueError(
            f"No columns found containing the string '{question_substring}'"
        )
    # Extract the relevant data
    filtered_data = data[matching_column].dropna()
    # Calculate histogram
    histogram_counts, bin_edges = np.histogram(
        filtered_data, bins=n_bins, range=bin_range
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    normalized_counts = histogram_counts / histogram_counts.sum()
    # Prepare data for chart
    chart_data = pd.DataFrame(
        {
            "Predicted Probability": bin_centers,
            "Percentage of Predictions": normalized_counts,
        }
    )
    chart = (
        alt.Chart(chart_data, height=350)
        .mark_bar(opacity=1.0)
        .encode(
            x=alt.X(
                "Predicted Probability:Q",
                bin=alt.Bin(maxbins=n_bins, extent=bin_range),
                title="Predicted probability",
            ),
            y=alt.Y(
                "Percentage of Predictions:Q",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 0.8]),
                title=f"% of predictions (n={histogram_counts.sum()})",
            ),
        )
    )
    return chart


@st.cache_data
def create_question_response_histogram(
    questions_fraction_answered, n_bins=14, bin_range=(0.725, 1.0)
):
    # Create histogram data
    histogram_counts, bin_edges = np.histogram(
        questions_fraction_answered, bins=n_bins, range=bin_range
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    chart_data = pd.DataFrame(
        {
            "Predicted Probability": bin_centers,
            "Percentage of Predictions": histogram_counts,
        }
    )
    # Create chart
    questions_fraction_answered_chart = (
        alt.Chart(chart_data)
        .mark_bar(opacity=1.0)
        .encode(
            x=alt.X(
                "Predicted Probability:Q",
                bin=alt.Bin(maxbins=n_bins, extent=bin_range),
                title="Fraction of participants answering the question",
            ),
            y=alt.Y(
                "Percentage of Predictions:Q",
                axis=alt.Axis(),
                scale=alt.Scale(),
                title="Numer of questions",
            ),
        )
        .properties(title=f"Response rate by question (n={histogram_counts.sum()})")
    )
    return questions_fraction_answered_chart


@st.cache_data
def create_participant_response_histogram(
    questions_fraction_answered, n_bins=20, bin_range=(0.0, 1.0)
):
    # Create histogram data
    histogram_counts, bin_edges = np.histogram(
        questions_fraction_answered, bins=n_bins, range=bin_range
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    chart_data = pd.DataFrame(
        {
            "Predicted Probability": bin_centers,
            "Percentage of Predictions": histogram_counts,
        }
    )
    # Create chart
    questions_fraction_answered_chart = (
        alt.Chart(chart_data)
        .mark_bar(opacity=1.0)
        .encode(
            x=alt.X(
                "Predicted Probability:Q",
                bin=alt.Bin(maxbins=n_bins, extent=bin_range),
                title="Fraction of questions predicted by participant",
            ),
            y=alt.Y(
                "Percentage of Predictions:Q",
                axis=alt.Axis(),
                scale=alt.Scale(),
                title="Numer of participants",
            ),
        )
        .properties(title=f"Response rate by particiapant (n={histogram_counts.sum()})")
    )
    return questions_fraction_answered_chart


def display_page_cards():
    # First row of cards
    card_column_1 = st.columns(2)
    with card_column_1[0]:
        card(
            title="Predictions by experience",
            text="Does self-reported forecasting experience correlate\
                with predictions?",
            image="http://placekitten.com/300/250",
            on_click=lambda: switch_page("Aggregating_predictions"),
            styles={
                "card": {
                    "margin-top": "-25px",
                    "margin-bottom": "-25px",
                }
            },
        )
    with card_column_1[1]:
        card(
            title="Aggregating predictions",
            text="""Can we aggregate Blind Mode predictions\
                to generate more accurate predictions?""",
            image="http://placekitten.com/301/250",
            on_click=lambda: switch_page("Aggregating_predictions"),
            styles={
                "card": {
                    "margin-top": "-25px",
                    "margin-bottom": "-25px",
                }
            },
        )
    # Second row of cards
    card_column_2 = st.columns(2)
    with card_column_2[0]:
        card(
            title="Simulating outcomes",
            text="""How do our aggregate predictions fare in Monte Carlo simulations of\
                possible futures?""",
            image="http://placekitten.com/300/251",
            on_click=lambda: switch_page("Simulating_outcomes"),
            styles={
                "card": {
                    "margin-top": "-25px",
                    "margin-bottom": "-25px",
                }
            },
        )
    with card_column_2[1]:
        card(
            title="Manifold evaluation",
            text="""How well do our aggregate predictions compare to prediction\
                markets across the year?""",
            image="http://placekitten.com/301/251",
            on_click=lambda: switch_page("Manifold_evaluation"),
            styles={
                "card": {
                    "margin-top": "-25px",
                    "margin-bottom": "-25px",
                }
            },
        )
    # Third row of cards, single wide card
    card(
        title="Post-hoc aggregation",
        text="""What attributes of Blind Mode participants are most predictive of\
            accurate forecasting?""",
        image="http://placekitten.com/400/252",
        on_click=lambda: switch_page("Post-hoc_aggregation"),
        styles={
            "card": {
                "width": "105%",  # <- % width of its container
                "margin-top": "-25px",
                "margin-bottom": "-25px",
            }
        },
    )
    return


def main():
    """Main script for Home.py

    Side effects:
        st.session_state.blind_mode_df is created
    """

    # Page configuration
    configure_page(page_title="ACX 2023 Prediction Contest")

    # Title and description of project
    st.title("ACX 2023 Prediction Contest")
    code_url = "https://github.com/jbreffle/acx-prediction-contest"
    st.markdown(
        f"""
        This is a streamlit app to explore the predictions made by participants in the
        [ACX 2023 Prediction Contest\
            ](<https://www.astralcodexten.com/p/2023-prediction-contest>).
        
        The contest included both a 
        ["Blind Mode"](<https://www.astralcodexten.com/p/2023-prediction-contest>)
        and a
        ["Full Mode"](<https://www.astralcodexten.com/p/stage-2-of-prediction-contest>).
        This app allows you to explore the predictions made by Blind Mode participants
        and demonstrates my approach to aggregating the data from the Blind Mode
        participants to generate predictions for the Full Mode.

        The app can be found at
        [jbreffle.github.io/acx-app](<https://jbreffle.github.io/acx-app>), 
        and the code for the app is at
        [github.com/jbreffle/acx-prediction-contest]({code_url}).
        """
    )
    # Describe app pages
    st.markdown(
        """
        This app has multiple pages, each of which address different questions:
        - **Home** (this page):
        What are some basic properties of the Blind Mode
        predictions?
        - **Predictions by experience**: 
        Does self-reported forecasting experience correlate with any property
        of the Blind Mode predictions?
        - **Aggregating predictions**:
        Can we aggregate Blind Mode predictions to generate more accurate predictions?
        - **Simulating outcomes**:
        How do our aggregated predictions fare in Monte Carlo simulations of
        possible futures?
        - **Prediction markets**:
        How well do our aggregate predictions compare to prediction markets
        across the year?
        - **Post-hoc aggregation**:
        How close to optimal were our aggregation parameters?
        - **Supervised aggregation**:
        What attributes of Blind Mode participants are most predictive of
        accurate forecasting?
        """
    )
    st.markdown(
        """
        Note:
        The contest has now concluded and the results were announced
        [here](<https://www.astralcodexten.com/p/who-predicted-2023>).
        """
    )
    st.divider()

    # Load and display data
    st.subheader("Blind Mode raw data")
    st.markdown(
        """
        The full dataset of the Blind Mode contest includes each participant's
        predictions for each of the 50 questions and their self-reported
        forecasting experience.
        The data also includes responses to a large set of survey questions for
        the subset of participants who chose to complete them.
        """
    )
    data_load_state = st.warning("Loading data...")
    blind_mode_df, markets_df, resolution_vector = load_data()
    data_load_state.success("Loading data... Done!")
    with st.expander("View raw data", expanded=False):
        st.write(blind_mode_df)
    st.divider()

    # Slider to select question to plot
    st.subheader("Prediction distributions")
    st.markdown(
        """
        Use the slider to select one of the 50 questions of the contest.
        The histogram will show the distribution of all Blind Mode participant
        predictions for that question.
        Now that the contest is over, we can also label the outcome of each question.
        """
    )
    selected_question_number = st.slider("Select question", 1, 50, 1)
    # Write question text and outcome
    full_question = get_question_text(markets_df, selected_question_number)
    question_outcome_string = get_question_outcome_string(
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
    # Prediction histogram for selected question
    question_histogram_chart = create_prediction_histogram(
        blind_mode_df, selected_question_number
    )
    st.altair_chart(question_histogram_chart, use_container_width=True)
    st.divider()

    # Links to the other pages of the app
    st.subheader("Page links")
    st.markdown(
        """Click a card below to go to the page,
        or use the links in the sidebar.
        """
    )
    display_page_cards()
    st.divider()

    # Response rates
    st.subheader("Appendix: Response rates")
    st.markdown(
        """
        Participants did not have to answer all questions.
        If you left a question blank, you would receive the average score
        for that question.
        Is there anything interesting regarding the response rates when
        analyzed by question or by participant?
        """
    )
    st.text("")
    # Response rate by question
    questions_fraction_answered = (
        1 - (blind_mode_df.filter(like="@", axis=1).isna().mean(axis=0))
    ).sort_values()
    questions_fraction_answered_chart = create_question_response_histogram(
        questions_fraction_answered
    )
    st.altair_chart(questions_fraction_answered_chart, use_container_width=True)
    st.markdown(
        """
        Most questions had a high resposne rate.
        Two questions are slight outliers, with low response rates.
        They are:
        """
    )
    questions_fraction_answered_numbers = questions_fraction_answered.index.str.extract(
        r"@(\d+)\."
    )
    full_question_1 = get_question_text(
        markets_df, questions_fraction_answered_numbers.iloc[0].values
    )
    full_question_2 = get_question_text(
        markets_df, questions_fraction_answered_numbers.iloc[1].values
    )
    st.markdown(
        f"""
        - Question {full_question_1} Which had a response rate of
        {questions_fraction_answered[0]:.02%}
        - Question {full_question_2} Which had a response rate of
        {questions_fraction_answered[1]:.02%}
        """
    )
    st.markdown(
        """
        This result makes sense, as both of these questions are two of the most
        obscure and niche-topic questions in the contest.
        """
    )
    st.text("")
    # Response rate by participant
    questions_fraction_answered = 1 - (
        blind_mode_df.filter(like="@", axis=1).isna().mean(axis=1)
    )
    questions_fraction_answered_chart = create_participant_response_histogram(
        questions_fraction_answered
    )
    st.altair_chart(questions_fraction_answered_chart, use_container_width=True)
    st.markdown(
        """
        Most participants answered most questions,
        many participants answered all questions,
        and a few participants answered zero questions.
        """
    )
    st.markdown(
        f"""
        - {(questions_fraction_answered==1).mean():.0%} of participants
        answered all questions.
        - {(questions_fraction_answered==0).sum()} participants answered zero questions.
        """
    )
    return


if __name__ == "__main__":
    main()
