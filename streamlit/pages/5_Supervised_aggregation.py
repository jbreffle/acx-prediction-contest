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


# Functions for page
@st.cache_data
def plot_feature_scatter(feature_df, blind_mode_brier_scores, feature_to_plot):
    y_scale = [0, 0.65]
    x_min = np.min(feature_df[feature_to_plot])
    x_max = np.max(feature_df[feature_to_plot])
    x_range = x_max - x_min
    x_scale = [x_min - 0.035 * (x_range), x_max + 0.035 * (x_range)]
    source = pd.DataFrame(
        {
            "feature": feature_df[feature_to_plot],
            "brier_score": blind_mode_brier_scores,
        }
    )
    chart = (
        alt.Chart(source)
        .mark_circle(size=60)
        .encode(
            x=alt.X("feature", title=feature_to_plot, scale=alt.Scale(domain=x_scale)),
            y=alt.Y(
                "brier_score", title="Brier score", scale=alt.Scale(domain=y_scale)
            ),
            tooltip=["feature", "brier_score"],
        )
        .interactive()
    )
    return chart


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
    st.write(
        """# What attributes of Blind Mode participants are most predictive of\
            accurate forecasting?"""
    )
    st.markdown(
        """
        Now that the contest is over, we can evaluate which of the blind mode 
        participants performed the best.
        We can then ask if there are characteristics of the best performing participants
        that we can identify and then use to generate an improved aggregate forecast.
        """
    )
    st.divider()

    # Plot any feature against score, scatter for continuous and violin for discrete
    st.subheader("Feature visualization")
    # Button randomly selects feature to plot
    feature_names = np.sort(feature_df.columns.values)
    option_default_index = int(np.where(feature_names == "Income")[0][0])
    feature_button = st.button("Randomize feature")
    if feature_button:
        option_default_index = np.random.randint(0, len(feature_names))
    feature_to_plot = st.selectbox(
        "Select a feature to plot", (feature_names), index=option_default_index
    )
    #
    # TODO replace pyplot with altair plot
    use_plt_fig = True
    if use_plt_fig:
        fig = plt.figure(figsize=(6, 3))
        ax, ax_histx, ax_histy = plot.feature_scatter_hist(
            feature_df, blind_mode_brier_scores, feature_to_plot, fig=fig
        )
        ax_histx.set_title("")
        st.pyplot(
            fig,
            use_container_width=True,
            transparent=True,
        )
    else:
        chart = plot_feature_scatter(
            feature_df, blind_mode_brier_scores, feature_to_plot
        )
        st.altair_chart(chart, use_container_width=True)
    st.divider()

    # Volcano plot of feature correlations with score
    st.subheader("Feature correlations")
    st.markdown(
        """
        We can also look at the correlation between each feature and the score.
        This can be done using a volcano plot, which shows the correlation coefficient
        on the x-axis and the p-value on the y-axis.
        """
    )
    # TODO replace pyplot with altair plot
    r_and_p_values = util.correlate_features_to_score(
        feature_df, blind_mode_brier_scores
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.feature_volcano_plot(r_and_p_values, ax=fig.gca()).set_title("")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )
    columns_cor = st.columns([1, 2.6, 1])
    with columns_cor[1]:
        st.dataframe(r_and_p_values.sort_values("r_value"), height=400)
    st.divider()

    # Show XGBoost model
    # TODO show XGBoost model:
    # model performance, feature importance
    st.subheader("XGBoost model")
    nb_xgb_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_xgb.ipynb"
    st.markdown(
        f"""
        See 
        [`./notebooks/5_post_hoc_xgb.ipynb`](<{nb_xgb_url}>)
        for ongoing analysis and results that will be transfered here,
        where an XGBoost model is used to predict a participants' Brier score based 
        on their survey question answers.
        """
    )
    st.divider()

    # Show aggregation method using model
    # TODO show aggregation method using model
    st.subheader("Aggregation from model")
    st.markdown(
        """
        The trained model can then be used to perform prediction aggregation on
        participants that were not part of the training set.
        """
    )
    st.divider()

    # Show NN model and aggregation method
    # TODO show NN model:
    st.subheader("Neural network model")
    ng_nn_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_nn.ipynb"
    st.markdown(
        f"""
        See 
        [`./notebooks/5_post_hoc_nn.ipynb`](<{ng_nn_url}>)
        for results similar to the XGBoost model, but using a neural network.
        """
    )
    # model performance, feature importance
    st.divider()

    # Footer
    nn_features_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_features.ipynb"
    st.markdown(
        f"""
        See 
        [`./notebooks/5_post_hoc_features.ipynb`](<{nn_features_url}>)
        for addidional analysis and results.
        """
    )

    return


if __name__ == "__main__":
    main()
