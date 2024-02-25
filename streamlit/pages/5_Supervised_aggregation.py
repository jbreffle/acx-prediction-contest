# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb


from src import process
from src import util
from src import plot
from src import models
import Home


# Functions for page
@st.cache_data
def plot_feature_correlations(feature_df, blind_mode_brier_scores):
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
    return None


@st.cache_data
def plot_feature_scatter_hist(feature_df, blind_mode_brier_scores, feature_to_plot):
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
    return None


@st.cache_data(show_spinner=False)
def train_xgboost_model(X_train, y_train, random_seed=42):
    # Create the xgboost model
    params = {
        "alpha": 0.3566354356611679,
        "gamma": 0.005997467172071246,
        "lambda": 0.370098278321425,
        "learning_rate": 0.1982493944040046,
        "min_child_weight": 3.310164889985418,
    }
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=random_seed, **params
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


@st.cache_data
def process_data_for_model(
    feature_df, estimates_matrix, blind_mode_brier_scores, random_seed=1337
):
    X, y, ests = models.prepare_data(
        feature_df, estimates_matrix, blind_mode_brier_scores
    )
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_seed, test_size=0.2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_seed, test_size=0.2
    )
    return X_train, X_test, y_train, y_test


@st.cache_data
def get_ml_dfs(df):
    feature_df = process.get_feature_df(df)
    target_df = process.get_target_df()
    return feature_df, target_df


def main():
    # Configure page
    Home.configure_page(page_title="Post-hoc aggregation")

    # Load data
    blind_mode_df, _, _ = Home.load_data()
    feature_df, target_df = get_ml_dfs(blind_mode_df)
    estimates_matrix = Home.get_estimates_matrix(blind_mode_df)
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
    feature_names = np.sort(feature_df.columns.values)
    option_default_index = int(np.where(feature_names == "Income")[0][0])
    feature_button = st.button("Randomize feature")
    if feature_button:  # Button randomly selects feature to plot
        option_default_index = np.random.randint(0, len(feature_names))
    feature_to_plot = st.selectbox(
        "Select a feature to plot", (feature_names), index=option_default_index
    )
    plot_feature_scatter_hist(feature_df, blind_mode_brier_scores, feature_to_plot)
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
    plot_feature_correlations(feature_df, blind_mode_brier_scores)
    st.divider()

    # Show XGBoost model: model performance, feature importance
    st.subheader("XGBoost model")
    # TODO finish XGBoost model
    if 0:
        X_train, X_test, y_train, y_test = process_data_for_model(
            feature_df, estimates_matrix, blind_mode_brier_scores
        )
        with st.spinner("Training XGBoost model..."):
            xgb_model = train_xgboost_model(X_train, y_train)
        # Train data
        y_train_pred = xgb_model.predict(X_train)
        st.write("Train data")
        fig = plt.figure(figsize=(6, 3))
        ax = plot.predicted_actual_scatter(y_train, y_train_pred)
        st.pyplot(fig, use_container_width=True, transparent=True)
        # Test data
        y_pred_test = xgb_model.predict(X_test)
        st.write("Test data")
        fig = plt.figure(figsize=(6, 3))
        ax = plot.predicted_actual_scatter(y_test, y_pred_test)
        st.pyplot(fig, use_container_width=True, transparent=True)
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
    st.subheader("Aggregation from model")
    st.markdown(
        """
        The trained model can then be used to perform prediction aggregation on
        participants that were not part of the training set.
        """
    )
    # TODO show aggregation method using model
    st.divider()

    # Show NN model and aggregation method
    st.subheader("Neural network model")
    # TODO show NN model: model performance, feature importance
    ng_nn_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_nn.ipynb"
    st.markdown(
        f"""
        See 
        [`./notebooks/5_post_hoc_nn.ipynb`](<{ng_nn_url}>)
        for results similar to the XGBoost model, but using a neural network.
        """
    )
    st.divider()

    # Link to notebooks
    nn_features_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_features.ipynb"
    st.markdown(
        f"""
        See 
        [`./notebooks/5_post_hoc_features.ipynb`](<{nn_features_url}>)
        for addidional analysis and results.
        """
    )
    opt_xgb_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/6_hyperopt_xgb.ipynb"
    opt_nn_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/6_hyperopt_nn.ipynb"
    st.markdown(
        f"""
        See 
        [`./notebooks/6_hyperopt_xgb.ipynb`](<{opt_xgb_url}>)
        and 
        [`./notebooks/6_hyperopt_nn.ipynb`](<{opt_nn_url}>)
        for hyperparameter optimization of the models.
        """
    )

    return


if __name__ == "__main__":
    main()
