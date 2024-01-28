# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd

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
