# pylint: disable=invalid-name
"""Streamlit app page
"""

# Imports
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE

from src import process
from src import util
from src import plot
import Home


@st.cache_data(show_spinner=False)
def run_tsne(parameter_mesh, exclude_group=None, n_components=None, random_state=0):
    # Create a list of the parameter_mesh columns to exclude
    parameter_columns = ["all", "sf", "fe", "lw", "beta_a", "beta_b"]
    if exclude_group is not None:
        # Match the index of the exclude_group to the parameter_columns and remove it
        if exclude_group in parameter_columns:
            exclude_index = parameter_columns.index(exclude_group)
            parameter_mesh = np.delete(parameter_mesh, exclude_index, axis=1)
        else:
            raise ValueError(f"exclude_group must be one of {parameter_columns}")

    if n_components is None and exclude_group is None:
        n_components = 2
    elif n_components is None:
        n_components = 1
    tsne = TSNE(n_components=n_components, random_state=random_state)
    tsne_weights = tsne.fit_transform(parameter_mesh)
    return tsne_weights


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
        add_jitter=True,
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
    return score_vec, parameter_mesh


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
    resolution_vector = target_df["resolution"].values
    return feature_df, target_df, resolution_vector


def main():
    # Configure page
    Home.configure_page(page_title="Post-hoc aggregation")

    # Load data
    blind_mode_df, _, _ = Home.load_data()
    _, _, resolution_vector = get_ml_dfs(blind_mode_df)

    # Introduction to page
    st.write("""# How close to optimal were our aggregation parameters""")
    st.markdown(
        """
        Now that the contest is over, we can evaluate how our aggregated predictions
        would have fared with alternate aggregation parameters.
        """
    )
    st.divider()

    # 1-D varied beta parameters
    st.subheader("Fixed weights and varying beta parameters")
    st.markdown(
        r"""
        This plot shows the Brier score of the aggregated predictions
        as a function of the beta parameters, with the group weights held constant
        at their original values and the two beta parameters equal to each other.
        Our chosen beta parameters of $$\beta _a = \beta_b = \frac{1}{3}$$
        turns out to have been close to optimal.
        (Note that our aggregation analysis used the `scipy.stats.beta.ppf` function,
        which makes the parameter values not directly comparable to those referenced in
        [Hanea et al., 2021](<https://doi.org/10.1371/journal.pone.0256919>))
        """
    )
    n_beta_range = 50
    beta_range, score_vec = calculate_score_vec_1d_beta(
        blind_mode_df, resolution_vector, n_beta_range=n_beta_range, equal_betas=True
    )
    # TODO: move to cached function
    fig = plt.figure(figsize=(6, 2))
    _ = plot.score_vs_beta(beta_range, score_vec).set_title("")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )

    # 2-D varied beta parameters
    st.markdown(
        """
        Same as above, but now allowing the beta parameters to vary independently.
        We see that a modest improvement in the Brier score is possible by
        allowing the beta parameters to be slightly different from each other.
        """
    )
    beta_range, score_vec = calculate_score_vec_1d_beta(
        blind_mode_df, resolution_vector, n_beta_range=n_beta_range, equal_betas=False
    )
    # TODO: move to cached function
    score_grid = score_vec.reshape((len(beta_range), len(beta_range)))
    fig = plt.figure(figsize=(6, 4))
    _ = plot.score_vs_beta_2d(beta_range, score_grid).set_title("")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )
    st.divider()

    # Fixed beta and varied weights
    st.subheader("Fixed beta parameters and varying weights")
    st.write(
        r"""
        The above analysis allowed the beta parameters to vary,
        but held the group weights.
        We can also hold the beta parameters constant and vary the group weights.

        For the following figures we fixed $$\beta _a = \beta_b = \frac{1}{3}$$ and
        varied the group weights across a 4D grid,
        allowing the weights to vary independently from 0 to 1 but considering only 
        parameter sets where the sum of the weights is 1.

        We see that for fixed beta parameters the Brier score does not
        vary much with the weights.
        This is consistent with our earlier finding that the mean predictions
        do not vary much between groups.
        """
    )
    score_vec, parameter_mesh = calculate_score_vec_all_params(
        blind_mode_df, resolution_vector, n_points=20, fixed_betas=True
    )
    # TODO: move to cached function
    fig = plt.figure(figsize=(6, 2))
    ax = plot.score_vec_hist(score_vec)
    ax.set_title("All Brier scores across the 4D grid of weights")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )

    # t-SNE
    st.write(
        """
        Dimensionality reduction of the 4d grid of weights using t-SNE.
        The Brier score of each parameter set is represented by the color of the point.
        Notice that the high Brier scores are clustered together,
        rather than spread out across the space.
        """
    )
    t_sne_seed = 0
    with st.spinner("Running t-SNE..."):
        tsne_weights_all = run_tsne(parameter_mesh, random_state=t_sne_seed)
        tsne_weights_minus_sf = run_tsne(
            parameter_mesh, exclude_group="sf", random_state=t_sne_seed
        )
        tsne_weights_minus_fe = run_tsne(
            parameter_mesh, exclude_group="fe", random_state=t_sne_seed
        )

    # 2D TSNE of parameter_mesh, colored by f
    # TODO: move to cached function
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.scatter(
        tsne_weights_all[:, 0],
        tsne_weights_all[:, 1],
        s=5,
        c=score_vec,
        cmap=mpl.colormaps["viridis"],
    )
    ax.set_xlabel("TSNE 1")
    ax.set_ylabel("TSNE 2")
    plt.colorbar(im, ax=ax, label="Brier score")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )

    # 1D TSNE without a dimension of parameter_mesh, then plot
    # that dimension against the 1-d TSNE
    st.markdown(
        """
        1D t-SNE of the parameter mesh when excluding one of the group weights
        (SF, left, and FE, right).
        Notice that when plotting the t-SNE against the weights of the Super 
        forecaster (SF) group weights, the hightest Brier scores are all
        associated with the highest SF weights. 
        This is not the case for the Forecasting experience (FE) group, 
        where the highest Brier scores are associated with the lowest FE weights.
        """
    )
    # TODO: move to cached function
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(121)
    ax.scatter(
        parameter_mesh[:, 1],
        tsne_weights_minus_sf[:, 0],
        s=3,
        c=score_vec,
        cmap="viridis",
    )
    ax.set_xlabel("SF weights")
    ax.set_ylabel("TSNE 1")
    ax = fig.add_subplot(122)
    ax.scatter(
        parameter_mesh[:, 2],
        tsne_weights_minus_fe[:, 0],
        s=3,
        c=score_vec,
        cmap="viridis",
    )
    ax.set_xlabel("FE weights")
    st.pyplot(
        fig,
        use_container_width=True,
        transparent=True,
    )
    st.divider()

    # Links to notebooks
    nb_4_url = "https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/4_post_hoc_aggregation.ipynb"
    st.markdown(
        f"""
        Click here 
        [`./notebooks/4_post_hoc_aggregation.ipynb`](<{nb_4_url}>)
        to see additional results,
        including dimensionality reduction with PCA and 
        global parameter optimization.
        By optimizing over the entire parameter space, we can find a minimal Brier Score
        of `0.147730` with fitted parameters of 
        `0.000000, 0.722385, 0.082014, 0.195602, 0.227142, 0.276849`.
        """
    )
    supervised_url = (
        "https://acx-prediction-contest.streamlit.app/Supervised_aggregation"
    )
    st.markdown(
        f"""
        Note that all of the analyses on this page result from evaluating the
        Brier score directly on the full data set.
        See the [Supervised aggregation](<{supervised_url}>) page for several 
        approaches to trainining a predictive model on the data.
        """
    )

    return


if __name__ == "__main__":
    main()
