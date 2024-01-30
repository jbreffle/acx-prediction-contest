# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt

from src import process
from src import util
from src import plot
import Home


@st.cache_data
def calculate_score_vec_all_params(
    blind_mode_df, resolution_vector, n_points=50, fixed_betas=True
):
    (
        _,
        mean_ests,
        mean_ests_SF,
        mean_ests_FE,
        mean_ests_LW,
    ) = get_aggregation_score_arguments(blind_mode_df)
    if fixed_betas:
        beta_range = [1 / 3]
    else:
        beta_range = np.linspace(1 / 5, 1 / 2, n_points)
    weights_range = np.linspace(0, 1, n_points)
    parameter_mesh = util.generate_aggregate_meshgrid(
        weights_range,
        weights_range,
        weights_range,
        beta_range,
        equal_betas=False,
    )
    # Compute score over the meshgrid
    score_vec = util.calculate_score_over_meshgrid(
        parameter_mesh,
        mean_ests,
        mean_ests_SF,
        mean_ests_FE,
        mean_ests_LW,
        resolution_vector,
    )
    return score_vec


@st.cache_data
def calculate_score_vec_1d_beta(
    blind_mode_df, resolution_vector, n_beta_range=50, equal_betas=True
):
    (
        weights_original,
        mean_ests,
        mean_ests_SF,
        mean_ests_FE,
        mean_ests_LW,
    ) = get_aggregation_score_arguments(blind_mode_df)
    beta_range = np.logspace(-2, 2, n_beta_range)
    parameter_mesh = util.generate_aggregate_meshgrid(
        *weights_original[1:], beta_range, equal_betas=equal_betas
    )
    score_vec = util.calculate_score_over_meshgrid(
        parameter_mesh,
        mean_ests,
        mean_ests_SF,
        mean_ests_FE,
        mean_ests_LW,
        resolution_vector,
    )
    return beta_range, score_vec


@st.cache_data
def get_aggregation_score_arguments(blind_mode_df):
    weights_original = [0.05, 0.8, 0.1, 0.05]
    estimates_df = blind_mode_df.filter(like="@", axis=1)
    mean_ests = np.array(estimates_df.mean(axis=0))
    mean_ests_SF = np.array(
        estimates_df.loc[blind_mode_df["Superforecaster"].values == "Yes"].mean(axis=0)
    )
    mean_ests_FE = np.array(
        estimates_df.loc[blind_mode_df["ForecastingExperience"].values == "Yes"].mean(
            axis=0
        )
    )
    mean_ests_LW = np.array(
        estimates_df.loc[blind_mode_df["LessWrong"].values == "Yes"].mean(axis=0)
    )
    return weights_original, mean_ests, mean_ests_SF, mean_ests_FE, mean_ests_LW


@st.cache_data
def get_ml_dfs(df):
    feature_df = process.get_feature_df(df)
    target_df = process.get_target_df()
    return feature_df, target_df


def main():
    # Configure page
    Home.configure_page(page_title="Post-hoc aggregation")

    # Load data
    blind_mode_df, markets_df, resolution_df = Home.load_data()
    estimates_df = blind_mode_df.filter(like="@", axis=1)
    feature_df, target_df = get_ml_dfs(blind_mode_df)
    estimates_matrix = process.get_estimates_matrix(blind_mode_df)
    resolution_vector = target_df["resolution"].values
    blind_mode_brier_scores = np.mean(
        np.square(estimates_matrix - resolution_vector), axis=1
    )

    # Introduction to page
    st.write("""# How close to optimal were our aggregation parameters""")
    st.markdown(
        """
        Now that the contest is over, we can evaluate how our aggregated predictions
        would have fared with alternate aggregation parameters.
        """
    )
    st.divider()

    st.subheader("Fixed weights and varying beta parameters")
    st.markdown("""1D grid:""")
    n_beta_range = 100
    beta_range, score_vec = calculate_score_vec_1d_beta(
        blind_mode_df, resolution_vector, n_beta_range=50, equal_betas=True
    )
    fig = plt.figure(figsize=(6, 3))
    _ = plot.score_vs_beta(beta_range, score_vec).set_title("")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )

    st.markdown("""2D grid:""")
    beta_range, score_vec = calculate_score_vec_1d_beta(
        blind_mode_df, resolution_vector, n_beta_range=50, equal_betas=False
    )
    score_grid = score_vec.reshape((len(beta_range), len(beta_range)))
    # Plot
    fig = plt.figure(figsize=(5, 5))
    _ = plot.score_vs_beta_2d(beta_range, score_grid).set_title("")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )

    st.write(
        r"""
        Varied group weights with fixed $$\beta _a = \beta_b = \frac{1}{3}$$
        """
    )
    score_vec = calculate_score_vec_all_params(
        blind_mode_df, resolution_vector, n_points=20, fixed_betas=True
    )
    fig = plt.figure(figsize=(6, 2))
    # TODO move this into a src.plot function
    _ = plot.score_vec_hist(score_vec)
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )

    # Plot any feature
    # Footer
    st.markdown(
        """
        See ```./notebooks/4_post_hoc_aggregation.ipynb``` of this project's GitHub repo 
        for results that will be transfered here.
        """
    )

    return


if __name__ == "__main__":
    main()
