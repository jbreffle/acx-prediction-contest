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

    # TODO
    st.subheader("Fixed weights and varying beta parameters")
    weights_original = [0.05, 0.8, 0.1, 0.05]
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
    beta_range = np.logspace(-2, 2, 100)
    parameter_mesh = util.generate_aggregate_meshgrid(
        *weights_original[1:], beta_range, equal_betas=True
    )
    score_vec = util.calculate_score_over_meshgrid(
        parameter_mesh,
        mean_ests,
        mean_ests_SF,
        mean_ests_FE,
        mean_ests_LW,
        resolution_vector,
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.score_vs_beta(beta_range, score_vec).set_title("")
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
