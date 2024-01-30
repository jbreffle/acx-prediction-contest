"""Misc. helper functions for the project.
"""


# Imports
import os
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.special import logit
from scipy.special import expit
from scipy.stats import beta
from sklearn.metrics import mean_squared_error

from src import process


def calculate_score_over_meshgrid(
    parameter_mesh,
    mean_ests,
    mean_ests_SF,
    mean_ests_FE,
    mean_ests_LW,
    resolution_vector,
):
    """..."""
    score_vec = np.zeros(len(parameter_mesh))
    for i in range(len(parameter_mesh)):
        score_vec[i] = aggregation_score(
            parameter_mesh[i, :],
            mean_ests,
            mean_ests_SF,
            mean_ests_FE,
            mean_ests_LW,
            resolution_vector,
        )
    return score_vec


def generate_aggregate_meshgrid(
    range_sf, range_fe, range_lw, range_betas, equal_betas=False
):
    """
    Generates a flattened 6D mesh grid, with an option to make parameter 5 equal to parameter 6.

    Args:
        values1, values2, values3, values4, values5 (array-like): Lists of values for the first 5 parameters.
        equal_56 (bool): If True, sets parameter 6 equal to parameter 5.

    Returns:
        numpy.ndarray: A 2D array where each row is a unique combination of the 6 parameters.
    """
    # Create initial grid, without second beta or column mean_ests (all participants)
    grids = np.meshgrid(
        range_sf, range_fe, range_lw, range_betas, indexing="ij", sparse=False
    )
    # Add a column for second beta
    if equal_betas:
        grids_5_equal_6 = [grids[i] for i in range(3)] + [grids[3], grids[3]]
    else:
        grids_5_equal_6 = np.meshgrid(
            range_sf,
            range_fe,
            range_lw,
            range_betas,
            range_betas,
            indexing="ij",
            sparse=False,
        )
    # Add a column for mean_ests, which the value needed to sum weights to 1
    column_for_mean = 1 - np.sum(grids_5_equal_6[0:3], axis=0)
    grids_5_equal_6.append(column_for_mean)
    grids_5_equal_6 = grids_5_equal_6[-1:] + grids_5_equal_6[:-1]
    # Flatten the meshgrid to a table of parameter sets
    flattened_grid = np.array(grids_5_equal_6).reshape(6, -1).T
    # Remove rows where column_for_mean is negative
    flattened_grid = flattened_grid[flattened_grid[:, 0] >= 0]
    return flattened_grid


def validated_aggregation_score(
    params, mean_ests, mean_ests_SF, mean_ests_FE, mean_ests_LW, resolution_vector
) -> float:
    """Scipy.optimize.basinhopping doesn't obey constrains when taking a step,
    so we need to validate the parameters before calculating the score.
    """
    error_score = 1000
    weights = params[:4]
    # If any params are nan, return a high score
    if np.any(np.isnan(params)):
        return error_score
    # If sum of weights is not 1, return a high score
    if np.abs(np.sum(weights) - 1) > 1e-6:
        return error_score
    score = aggregation_score(
        params, mean_ests, mean_ests_SF, mean_ests_FE, mean_ests_LW, resolution_vector
    )
    return score


def aggregation_score(
    params, mean_ests, mean_ests_SF, mean_ests_FE, mean_ests_LW, resolution_vector
) -> float:
    # Unpack params
    weights = params[:4]
    beta_a = params[4]
    beta_b = params[5]
    # Calculate
    weighted_mean_predictions = (
        weights[0] * mean_ests
        + weights[1] * mean_ests_SF
        + weights[2] * mean_ests_FE
        + weights[3] * mean_ests_LW
    )
    weighted_tranformed_predictions = beta.ppf(
        weighted_mean_predictions / 100, beta_a, beta_b
    )
    resulting_predictions = np.round(weighted_tranformed_predictions * 100).clip(1, 99)
    result = mean_squared_error(
        resulting_predictions / 100,
        resolution_vector,
    )
    return result


def aggregation_constraint_1(params):
    weights_sum_to_one = params[0] + params[1] + params[2] + params[3] - 1
    return weights_sum_to_one


def correlate_features_to_score(feature_df, scores):
    """For each feature, calculate the correlation to the brier score"""
    r_values = []
    p_values = []
    for col in feature_df.columns:
        nan_mask = ~np.isnan(feature_df[col]) & ~np.isnan(scores)
        r_value, p_value = pearsonr(
            feature_df[col][nan_mask],
            scores[nan_mask],
        )
        r_values.append(r_value)
        p_values.append(p_value)
    r_and_p_values = pd.DataFrame(
        {"r_value": r_values, "p_value": p_values},
        index=feature_df.columns,
    )
    return r_and_p_values


def ordinal(n):
    return str(n) + (
        "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )


def softmax(vec, temperature):
    """Softmax with temperature parameter.

    Args:
        vec (list or numpy.ndarray): Input vector.
        temperature (float): Temperature parameter.

    Returns:
        numpy.ndarray: Softmax probabilities.
    """
    max_val = np.max(vec)  # Find the maximum value for normalization
    adjusted_vec = (vec - max_val) / temperature  # Adjust values with temperature
    exps = np.exp(adjusted_vec)  # Compute exponentials
    sum_exps = np.sum(exps)  # Sum of exponentials for normalization
    return exps / sum_exps


def sim_blind_mode_comparison(
    base_predictions,
    estimates_matrix,
    n_simulations=1000,
    simulation_noise=0.0,
    beta_a=1 / 1,
    beta_b=1 / 1,
):
    # with noise of 1, predictions are approximately randomly 0 or 1
    # with noise of 0.001, predictions are approximately the same
    base_predictions = beta.ppf(base_predictions, beta_a, beta_b)
    base_predictions = np.clip(base_predictions, 0, 1)
    # Run Monte Carlo simulation
    my_brier_score_percentile = []
    for _ in range(n_simulations):
        # Add noise to base predictions, use logit to transform to un-bounded space
        base_predictions = expit(
            logit(base_predictions)
            + np.random.normal(0, simulation_noise, base_predictions.shape)
        )
        # Simulate outcomes
        sim_outcomes = np.random.binomial(1, base_predictions)
        # Compute the Brier score percentile
        my_brier_score = np.mean((base_predictions - sim_outcomes) ** 2)
        blind_mode_scores = np.mean((estimates_matrix - sim_outcomes) ** 2, axis=1)
        my_brier_score_percentile.append(
            np.mean(blind_mode_scores < my_brier_score) * 100
        )
    return my_brier_score, blind_mode_scores, my_brier_score_percentile


def sim_binary_comparison(
    n_simulations, my_preds, base_preds, base_preds_are_probs=True
):
    """Calculates Bier scores from Monte Carlo simulation of forecast outcomes.

    Simulate outcomes based on the base predictions and compute the Brier score
    for the base predictions and my predictions.

    Args:
        n_simulations: number of simulations
        my_preds: my predictions, a numpy array of probabilities
        base_preds: base predictions, a numpy array of probabilities that are
            1) used for simulating outcomes (if base_preds_are_probs=True) and
            2) used as the baseline for comparison
        base_preds_are_probs: whether the base predictions are probabilities (as
            opposed to percentages)

    Returns:
        my_brier_scores: ...
        base_brier_scores: ...
        my_score_percentiles: ...
    """
    # Initialize outputs
    base_brier_scores = np.zeros(n_simulations)
    my_brier_scores = np.zeros(n_simulations)
    my_score_percentiles = np.zeros(n_simulations)
    if base_preds_are_probs:
        sim_preds = base_preds
    else:
        sim_preds = my_preds
    # Run the simulations
    for ith_sim in range(n_simulations):
        ith_sim_outcomes = np.random.binomial(1, sim_preds)
        base_brier_scores[ith_sim] = np.mean((base_preds - ith_sim_outcomes) ** 2)
        my_brier_scores[ith_sim] = np.mean((my_preds - ith_sim_outcomes) ** 2)
    # Compute the Brier score percentile. Note: lower Brier score is better
    for ith_sim in range(n_simulations):
        my_score_percentiles[ith_sim] = (
            np.sum(my_brier_scores[ith_sim] < base_brier_scores) / n_simulations * 100
        )
    return my_brier_scores, base_brier_scores, my_score_percentiles


def perform_frac_ne_bootstrap(
    df, ne, n_iterations=1000, seed=42, data_path=None, silent=False
):
    """
    Performs a bootstrap test for different ne values and saves the results.

    If results already exist for a given configuration, loads them instead.
    Filesnames are generated using the hash of the configuration string,
    based on the ne values, number of iterations, and random seed.

    Parameters:
    - df: DataFrame containing the predictions and class labels.
    - ne: List of values to be excluded in the calculation.
    - n_iterations: Number of bootstrap iterations.
    - seed: Random seed for reproducibility.
    - base_path: Base path to save the bootstrap results.

    Returns:
    - Dictionary of bootstrap distributions for each class.
    """
    if data_path is None:
        data_path = process.PROCESSED_DATA_FOLDER
    # Generate a unique hash for the configuration
    config_str = f"{str(ne)}_{n_iterations}_{seed}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:10]
    file_path = os.path.join(data_path, f"bootstrap_ne_predictions_{config_hash}.csv")
    # Check if the results file already exists
    if os.path.exists(file_path):
        if not silent:
            print(f"Loading bootstrap results from {file_path}")
        bootstrap_results = pd.read_csv(file_path)
        fractions_dist = {
            cls: bootstrap_results[cls].values for cls in bootstrap_results.columns
        }
        return fractions_dist
    np.random.seed(seed)
    # Initialize the distribution arrays
    fractions_dist = {cls: np.zeros(n_iterations) for cls in df["class"].unique()}
    # Perform the bootstrap process
    for cls in fractions_dist:
        class_predictions = df[df["class"] == cls]["prediction"]
        for i in range(n_iterations):
            trial = np.random.choice(class_predictions, size=len(class_predictions))
            trial_frac = len([x for x in trial if x not in ne]) / len(trial)
            fractions_dist[cls][i] = trial_frac
    # Save the results
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(fractions_dist).to_csv(file_path, index=False)
    if not silent:
        print(f"Saved bootstrap results to {file_path}")
    return fractions_dist


def calculate_not_equal_prediction_fractions(df, ne):
    classes = df["class"].unique()
    fractions = {
        cls: len(df[(df["class"] == cls) & (~df["prediction"].isin(ne))])
        / len(df[df["class"] == cls])
        for cls in classes
    }
    return fractions


def perform_ne_chi_square_test(frac1, count1, frac2, count2):
    table = [
        [frac1 * count1, (1 - frac1) * count1],
        [frac2 * count2, (1 - frac2) * count2],
    ]
    stat, p, _, _ = chi2_contingency(table)
    return stat, p


def print_fraction_predictions_not_rounded(df, round_digits=4):
    """
    Calculates and prints the fractions of predictions under certain conditions.

    Parameters:
    - df: DataFrame containing the predictions and class labels.
    - classes: List of classes to include in the calculation.
    - round_digits: Number of decimal places for rounding the output.
    """
    classes = df["class"].unique()
    max_class_name_length = max(len(class_name) for class_name in classes)
    condition_descriptions = [
        "Fraction of predictions at the extreme (1 or 99)",
        "Fraction of predictions that are not 1, 99, or divisible by 5",
        "Fraction of predictions that are not divisible by 5",
    ]
    conditions = [
        {
            "description": condition_descriptions[0],
            "condition": lambda x: x in [1, 99],
        },
        {
            "description": condition_descriptions[1],
            "condition": lambda x: x not in [1, 99] + list(range(5, 100, 5)),
        },
        {
            "description": condition_descriptions[2],
            "condition": lambda x: x not in list(range(5, 100, 5)),
        },
    ]
    for condition in conditions:
        print(condition["description"])
        for class_name in classes:
            class_predictions = df.loc[df["class"] == class_name, "prediction"]
            fraction = len(
                [x for x in class_predictions if condition["condition"](x)]
            ) / len(class_predictions)
            f_text_1 = class_name.rjust(max_class_name_length)
            f_text_2 = round(fraction, round_digits)
            print(f"{f_text_1}: {f_text_2}")
        print()
    return None
