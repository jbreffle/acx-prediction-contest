# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

from src import util
import Home


@st.cache_data
def plot_sim_outcome_dist(
    my_brier_scores, base_brier_scores, n_bins=50, bin_range=(0.1, 0.35)
):
    # Compute histogram data for base_brier_scores
    hist_base, bin_edges_base = np.histogram(
        base_brier_scores, bins=n_bins, range=bin_range, density=True
    )
    bin_centers_base = (bin_edges_base[:-1] + bin_edges_base[1:]) / 2.0
    # Compute histogram data for my_brier_scores
    hist_my, bin_edges_my = np.histogram(
        my_brier_scores, bins=n_bins, range=bin_range, density=True
    )
    bin_centers_my = (bin_edges_base[:-1] + bin_edges_base[1:]) / 2.0
    bin_step = bin_edges_base[1] - bin_edges_base[0]
    # Plot line chart, either bin centers or step chart
    plot_bin_center = False
    if plot_bin_center:
        # Combine data for plotting
        source = pd.DataFrame(
            {
                "Type": np.concatenate(
                    (
                        np.repeat("Base Brier Scores", len(hist_base)),
                        np.repeat("My Brier Scores", len(hist_my)),
                    )
                ),
                "Brier Score": np.concatenate((bin_centers_base, bin_centers_my)),
                "Probability Density": np.concatenate((hist_base, hist_my)),
            }
        )
        # Create the Altair chart
        chart = (
            alt.Chart(source, height=350)
            .mark_line(opacity=1.0, point=True)
            .encode(
                x=alt.X(
                    "Brier Score:Q", title="Brier Score", bin=alt.Bin(maxbins=n_bins)
                ),
                y=alt.Y("Probability Density:Q", title="Probability Density"),
                color=alt.Color(
                    "Type:N", legend=alt.Legend(direction="horizontal", orient="top")
                ),
            )
        )
    else:
        # Repeat each bin edge twice (including the first and last edges)
        step_x_my = np.repeat(bin_edges_my, 2)
        step_x_base = np.repeat(bin_edges_base, 2)
        # Create step_y with repeated histogram heights and zeros interleaved
        step_y_my = np.zeros(len(step_x_my))
        step_y_my[1:-1] = np.repeat(hist_my, 2)
        step_y_base = np.zeros(len(step_x_base))
        step_y_base[1:-1] = np.repeat(hist_base, 2)
        # Combine data for plotting
        source = pd.DataFrame(
            {
                "Type": np.concatenate(
                    (
                        np.repeat("Base Brier Scores", len(step_x_base)),
                        np.repeat("My Brier Scores", len(step_x_my)),
                    )
                ),
                "Brier Score": np.concatenate((step_x_base, step_x_my)),
                "Probability Density": np.concatenate((step_y_base, step_y_my)),
            }
        )
        # Create the Altair chart
        # Only plot the outline, not the fill of the mark_bar
        chart = (
            alt.Chart(source, height=350)
            .mark_line()  # Changed to mark_area with outline
            .encode(
                x=alt.X(
                    "Brier Score:Q",
                    title="Brier Score",
                    bin=alt.Bin(maxbins=n_bins, minstep=bin_step, extent=bin_range),
                ),
                y=alt.Y("Probability Density", title="Probability Density"),
                color=alt.Color(
                    "Type:N", legend=alt.Legend(direction="horizontal", orient="top")
                ),
            )
        )
    return chart


@st.cache_data
def plot_sim_outcome_prct_dist(my_score_percentiles, n_bins=20):
    counts, bin_edges = np.histogram(my_score_percentiles, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # Prepare data for chart
    source = pd.DataFrame(
        {"My Brier score percentile": bin_centers, "Probability density": counts}
    )
    # Create the histogram chart
    chart = (
        alt.Chart(source, height=300)
        .mark_bar(opacity=1.0)
        .encode(
            alt.X(
                "My Brier score percentile:Q",
                bin=alt.Bin(maxbins=n_bins),
                title="My Brier score percentile",
            ),
            alt.Y("Probability density:Q", title="Probability density"),
        )
    )
    return chart


@st.cache_data
def plot_blind_mode_histogram(blind_mode_scores, my_brier_score, n_bins=20):
    bin_range = [0.10, 0.55]
    bin_range[0] = min(bin_range[0], np.min(blind_mode_scores))
    bin_range[1] = max(bin_range[1], np.max(blind_mode_scores))
    # Calculate histogram counts and edges
    counts, bin_edges = np.histogram(blind_mode_scores, bins=n_bins, range=bin_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_step = bin_edges[1] - bin_edges[0]
    # Prepare data for chart
    source = pd.DataFrame({"Blind Mode Scores": bin_centers, "Count": counts})
    chart = (
        alt.Chart(source, height=300)
        .mark_bar(opacity=1.0)
        .encode(
            alt.X(
                "Blind Mode Scores:Q",
                bin=alt.Bin(maxbins=n_bins, extent=bin_range, step=bin_step),
                title="Brier score (lower is better)",
            ),
            alt.Y("Count:Q", title="Count (Blind Mode participants)"),
        )
    )
    # Create a line for my_brier_score
    vline = (
        alt.Chart(pd.DataFrame({"x": [my_brier_score]}))
        .mark_rule(color="red")
        .encode(x="x:Q")
    )
    # Add a legend, describing the red line
    legend = (
        alt.Chart(
            pd.DataFrame({"x": [my_brier_score], "label": ["Aggregate prediction"]})
        )
        .mark_text(color="red", align="center", dx=0, dy=-10, size=12)
        .encode(x="x:Q", y=alt.value(0), text="label:N")
    )

    return chart + vline + legend


@st.cache_data
def plot_brier_score_percentile_histogram(my_brier_score_percentile, n_bins=20):
    # Calculate histogram counts and edges
    counts, bin_edges = np.histogram(my_brier_score_percentile, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_step = bin_edges[1] - bin_edges[0]
    # Prepare data for chart
    source = pd.DataFrame({"Brier Score Percentile": bin_centers, "Count": counts})
    # Create the histogram
    chart = (
        alt.Chart(source, height=300)
        .mark_bar(opacity=1.0)
        .encode(
            alt.X(
                "Brier Score Percentile:Q",
                bin=alt.Bin(maxbins=n_bins, step=bin_step),
                title="Brier score percentile (lower is better)",
            ),
            alt.Y("Count:Q", title="Count (simulations)"),
        )
    )
    return chart


# Functions for page
@st.cache_data
def run_binary_comparison(
    my_predictions,
    sf_mean_predictions,
    n_simulations=1000,
    base_preds_are_probs=True,
    rng_seed=42,
):
    np.random.seed(rng_seed)
    (
        my_brier_scores,
        base_brier_scores,
        my_score_percentiles,
    ) = util.sim_binary_comparison(
        n_simulations,
        my_predictions,
        sf_mean_predictions,
        base_preds_are_probs=base_preds_are_probs,
    )
    return my_brier_scores, base_brier_scores, my_score_percentiles


@st.cache_data
def run_blind_mode_comparison(my_predictions, estimates_matrix, rng_seed=42, **args):
    """
    Arguments:
        base_predictions,
        estimates_matrix,
        n_simulations=1000,
        simulation_noise=0.0,
        beta_a=1 / 1,
        beta_b=1 / 1,
    """
    np.random.seed(rng_seed)
    (
        my_brier_score,
        blind_mode_scores,
        my_brier_score_percentile,
    ) = util.sim_blind_mode_comparison(my_predictions, estimates_matrix, **args)
    return my_brier_score, blind_mode_scores, my_brier_score_percentile


def main():
    # Configure page
    Home.configure_page(page_title="Simulating outcomes")

    # Load data
    blind_mode_df, _, _ = Home.load_data()
    my_predictions, sf_mean_predictions = Home.load_predictions()
    estimates_matrix = Home.get_estimates_matrix(blind_mode_df)

    # Introduction to page
    st.write(
        """# How do our aggregate predictions fare in Monte Carlo simulations of\
                possible futures?"""
    )
    st.markdown(
        r"""
        While we don't know the future, we can simulate it. 
        We can take some probaility of an event occuring and then simulate many possible 
        futures by taking that probabilty as the chance of the event occuring.
        We can then evaluate the distribution of our forecasting scores that might occur 
        due to the randomness of the future.

        The scoring method that will be used for the contest was not given,
        so we will use the Brier score,
        which is a common scoring method for binary predictions.
        The Brier score is defined as the mean squared difference between the predicted
        probability and the actual outcome,
        which means that a lower Brier score is better.

        This is given by the equation:
        $$
        Brier = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2
        $$
        where $f_i$ is the prediction probability and $o_i$ is the outcome for each
        of the $N$ events.
        """
    )
    st.divider()

    # Simulate outcomes, using Super forcaster predictions as probabilities
    st.subheader("Aggregated predictions vs the Super forecasters")
    st.markdown(
        "#### What if the mean Super forcaster predictions were correctly calibrated?"
    )
    st.markdown(
        """
        Here are simulation results from assuming that the mean predictions of the
        Super forecaster group are the true underlying probabilities of the events.
        Clicking the big red button will run new simulations.
        """
    )
    rng_seed_sf = st.session_state.get("rng_seed_sf", 42)
    if st.button("Run new superforcaster-mean simulations", type="primary"):
        rng_seed_sf = np.random.randint(0, 100000)
    st.session_state.rng_seed_sf = rng_seed_sf
    my_brier_scores, base_brier_scores, my_score_percentiles = run_binary_comparison(
        my_predictions,
        sf_mean_predictions,
        base_preds_are_probs=True,
        rng_seed=rng_seed_sf,
    )
    # Histogram of brier scores, Note: lower is better for brier scores
    fig = plot_sim_outcome_dist(my_brier_scores, base_brier_scores)
    st.altair_chart(fig, use_container_width=True)
    st.markdown(
        """
        The above distribution shows the Brier scores of the aggregate predictions
        and the mean Super forecaster predictions over many simulated futures.
        """
    )
    fig = plot_sim_outcome_prct_dist(my_score_percentiles)
    st.altair_chart(fig, use_container_width=True)
    median_brier_score_percentile = np.median(my_score_percentiles)
    st.markdown(
        f"""
        The median Brier score percentile of the aggreagate predictions
        is {median_brier_score_percentile:.2f}.
        """
    )

    # Simulate outcomes, using my predictions as probabilities
    st.markdown("#### What if the aggregated predictions had perfect calibration?")
    st.markdown(
        """Here are results from assuming that the aggregated predictions are the true
        underlying probabilities:
        """
    )
    rng_seed_my_preds = st.session_state.get("rng_seed_my_preds", 43)
    if st.button("Run new prefect-calibration simulations", type="primary"):
        rng_seed_my_preds = np.random.randint(0, 100000)
    st.session_state.rng_seed_my_preds = rng_seed_my_preds
    my_brier_scores, base_brier_scores, my_score_percentiles = run_binary_comparison(
        my_predictions,
        sf_mean_predictions,
        base_preds_are_probs=False,
        rng_seed=rng_seed_my_preds,
    )
    # Histogram of brier scores, note: lower is better for brier scores
    fig = plot_sim_outcome_dist(
        my_brier_scores, base_brier_scores, bin_range=(0.075, 0.275)
    )
    st.altair_chart(fig, use_container_width=True)
    fig = plot_sim_outcome_prct_dist(my_score_percentiles)
    st.altair_chart(fig, use_container_width=True)
    median_brier_score_percentile = np.median(my_score_percentiles)
    st.markdown(
        f"""
        The median Brier score percentile of the aggreagate predictions
        is {median_brier_score_percentile:.2f}.
        """
    )
    st.divider()

    # Comparing my predictions to all Blind Mode participants
    st.subheader("Aggregated predictions vs Blind Mode predictions")
    st.markdown(
        """How do the aggregated predictions fare against all Blind Mode participants' 
        predictions in Monte Carlo simulations of possible futures?
        """
    )
    rng_seed_blind_mode = st.session_state.get("rng_seed_blind_mode", 44)
    if st.button("Run new Blind Mode evaluation simulations", type="primary"):
        rng_seed_blind_mode = np.random.randint(0, 100000)
    st.session_state.rng_seed_blind_mode = rng_seed_blind_mode
    (
        my_brier_score,
        blind_mode_scores,
        my_brier_score_percentile,
    ) = run_blind_mode_comparison(
        my_predictions, estimates_matrix, rng_seed=rng_seed_blind_mode
    )
    fig = plot_blind_mode_histogram(blind_mode_scores, my_brier_score)
    st.altair_chart(fig, use_container_width=True)
    fig = plot_brier_score_percentile_histogram(my_brier_score_percentile)
    st.altair_chart(fig, use_container_width=True)
    mean_brier_score_percentile = np.mean(my_brier_score_percentile)
    min_percentile_to_win = 100 / len(blind_mode_scores)
    frac_wins = np.mean(np.array(my_brier_score_percentile) < min_percentile_to_win)
    st.markdown(
        f"""
        It turns out that even if the aggregated predictions were perfectly calibrated,
        our predictions still would not be likely to win even the Blind Mode 
        competition.
        The mean percentile of the aggregate predictions is
        {mean_brier_score_percentile:.2f}%.
        That is a strong result, but in a field of {len(blind_mode_scores)}
        participants, it is not enough to win.
        That percentile corresponds to finishing in 
        {util.ordinal(round(len(blind_mode_scores)*(mean_brier_score_percentile/100)))}
        place.
                
        In only {frac_wins*100}% of the simulations do the aggregate predictions win,
        even though we know that they are __*perfectly calibrated*__.
        """
    )

    return


if __name__ == "__main__":
    main()
