"""Build a static replica of the Streamlit app.

The generated files under dist/ are suitable for GitHub Pages or any static
host. Heavy Python work is performed here; the browser app only renders charts,
tables, and lightweight controls.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta
from sklearn.manifold import TSNE

from src import process
from src import util


ROOT = Path(__file__).resolve().parents[1]
STATIC_SRC = ROOT / "static_app"
DIST = ROOT / "dist"
DATA_DIR = DIST / "data"
FEATURE_DIR = DATA_DIR / "features"
MARKET_DIR = DATA_DIR / "markets"

GROUPS = {
    "all": "All participants",
    "FE": "Forcasting experience",
    "LW": "LessWrong member",
    "SF": "Super forecaster",
}


def clean_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def clean_array(values, digits: int | None = None):
    output = []
    for value in np.asarray(values, dtype=object).tolist():
        value = clean_scalar(value)
        if isinstance(value, float) and digits is not None:
            value = round(value, digits)
        output.append(value)
    return output


def table_payload(df: pd.DataFrame) -> dict:
    payload = json.loads(df.to_json(orient="split"))
    return {"columns": payload["columns"], "rows": payload["data"]}


def records_payload(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def reset_dist() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(STATIC_SRC, DIST)
    shutil.copy2(ROOT / "assets/images/favicons/icon.png", DIST / "assets/icon.png")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_DIR.mkdir(parents=True, exist_ok=True)


def get_question_number(column_name: str) -> int:
    match = re.search(r"@(\d+)\.", column_name)
    if not match:
        raise ValueError(f"Could not extract question number from {column_name}")
    return int(match.group(1))


def histogram_payload(values, bins=10, bin_range=None, normalize=False, density=False):
    values = pd.Series(values).dropna()
    counts, edges = np.histogram(values, bins=bins, range=bin_range, density=density)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()
    return {
        "x": clean_array(centers, 6),
        "y": clean_array(counts, 8),
        "edges": clean_array(edges, 6),
        "n": int(len(values)),
    }


def group_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=df.index),
        "FE": df["ForecastingExperience"].eq("Yes"),
        "LW": df["LessWrong"].eq("Yes"),
        "SF": df["Superforecaster"].eq("Yes"),
    }


def make_questions(markets_df: pd.DataFrame, resolution_df: pd.DataFrame) -> list[dict]:
    markets = markets_df.copy()
    markets["question_number"] = markets["question_number"].astype(int)
    markets = markets.sort_values("question_number")
    resolutions = resolution_df.set_index("question_number")["resolution"]
    questions = []
    for row in markets.itertuples(index=False):
        number = int(row.question_number)
        resolution = int(resolutions.loc[number])
        questions.append(
            {
                "number": number,
                "text": row.question,
                "outcome": "Yes" if resolution == 1 else "No",
                "resolution": resolution,
                "probability": clean_scalar(row.probability),
            }
        )
    return questions


def build_home(df: pd.DataFrame, questions: list[dict]) -> None:
    question_cols = sorted(df.filter(like="@").columns, key=get_question_number)
    prediction_histograms = {}
    for col in question_cols:
        q_number = get_question_number(col)
        prediction_histograms[str(q_number)] = histogram_payload(
            df[col], bins=10, bin_range=(0, 100), normalize=True
        )

    question_response = (1 - df[question_cols].isna().mean(axis=0)).sort_values()
    participant_response = 1 - df[question_cols].isna().mean(axis=1)
    payload = {
        "questions": questions,
        "predictionHistograms": prediction_histograms,
        "responseByQuestion": histogram_payload(
            question_response, bins=14, bin_range=(0.725, 1.0)
        ),
        "responseByParticipant": histogram_payload(
            participant_response, bins=20, bin_range=(0, 1)
        ),
        "responseSummary": {
            "allQuestionsPercent": round(float((participant_response == 1).mean() * 100)),
            "zeroQuestionsCount": int((participant_response == 0).sum()),
        },
    }
    write_json(DATA_DIR / "home.json", payload)
    write_json(DATA_DIR / "raw-data.json", table_payload(df))


def calculate_group_statistics(ests: pd.DataFrame, df: pd.DataFrame, statistic: str) -> pd.DataFrame:
    stat_function = getattr(pd.DataFrame, statistic)
    masks = group_masks(df)
    group_stats = pd.DataFrame()
    group_stats["All_Participants"] = stat_function(ests, axis=0).values
    group_stats["Forcasting_experience"] = stat_function(ests.loc[masks["FE"]], axis=0).values
    group_stats["LessWrong_member"] = stat_function(ests.loc[masks["LW"]], axis=0).values
    group_stats["Super forecaster"] = stat_function(ests.loc[masks["SF"]], axis=0).values
    return group_stats


def regression_payload(points: pd.DataFrame, x_col: str, y_col: str) -> dict:
    x = points[x_col].to_numpy(dtype=float)
    y = points[y_col].to_numpy(dtype=float)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    _, _, _, p_value, _ = stats.linregress(x, y - x)
    angle = np.rad2deg(np.arctan(slope))
    return {
        "line": {
            "x": [0, 100],
            "y": clean_array([intercept, intercept + slope * 100], 6),
        },
        "p": clean_scalar(p_value),
        "angle": clean_scalar(angle),
    }


def chi_square_for_ne(flattened_prediction_df: pd.DataFrame, ne: list[int]) -> tuple[pd.DataFrame, dict]:
    fractions = util.calculate_not_equal_prediction_fractions(flattened_prediction_df, ne)
    rows = []
    for cls in ["FE", "SF", "LW"]:
        stat, p_value = util.perform_ne_chi_square_test(
            fractions["all"],
            len(flattened_prediction_df[flattened_prediction_df["class"] == "all"]),
            fractions[cls],
            len(flattened_prediction_df[flattened_prediction_df["class"] == cls]),
        )
        rows.append({"class": cls, "stat": stat, "p": p_value})
    rows.append({"class": "all", "stat": np.nan, "p": np.nan})
    chi_square_results = pd.DataFrame(rows)
    chi_square_results["percent"] = [
        fractions[cls] * 100 for cls in chi_square_results["class"].unique()
    ]
    chi_square_results.sort_values(by="class", ascending=False, inplace=True)
    fractions_dist = util.perform_frac_ne_bootstrap(
        flattened_prediction_df, ne, n_iterations=1000, silent=True
    )
    return chi_square_results, fractions_dist


def bootstrap_histograms(fractions_dist: dict[str, np.ndarray]) -> dict:
    n_bins = int(round(np.sqrt(sum(len(values) for values in fractions_dist.values()))))
    min_value = min(float(np.min(values)) for values in fractions_dist.values())
    max_value = max(float(np.max(values)) for values in fractions_dist.values())
    payload = {}
    for key, values in fractions_dist.items():
        payload[key] = histogram_payload(values, bins=n_bins, bin_range=(min_value, max_value))
    return payload


def build_experience(df: pd.DataFrame) -> None:
    ests = df.filter(like="@")
    masks = group_masks(df)
    question_cols = sorted(ests.columns, key=get_question_number)
    segmented_histograms = {}
    metrics_by_question = {}

    mean_stats = calculate_group_statistics(ests, df, "mean")
    for col in question_cols:
        q_number = get_question_number(col)
        histograms = {}
        for key, mask in masks.items():
            histograms[key] = histogram_payload(
                df.loc[mask, col], bins=10, bin_range=(0, 100), normalize=True
            )
        segmented_histograms[str(q_number)] = histograms
        row = mean_stats.iloc[q_number - 1]
        metrics_by_question[str(q_number)] = {
            "all": clean_scalar(row["All_Participants"]),
            "FE": clean_scalar(row["Forcasting_experience"]),
            "LW": clean_scalar(row["LessWrong_member"]),
            "SF": clean_scalar(row["Super forecaster"]),
        }

    scatter = {}
    for statistic in ["median", "mean"]:
        stats_df = calculate_group_statistics(ests, df, statistic)
        scatter[statistic] = {}
        for key, y_col in [
            ("FE", "Forcasting_experience"),
            ("LW", "LessWrong_member"),
            ("SF", "Super forecaster"),
        ]:
            scatter[statistic][key] = {
                "points": records_payload(
                    stats_df[["All_Participants", y_col]].rename(
                        columns={"All_Participants": "x", y_col: "y"}
                    )
                ),
                "regression": regression_payload(stats_df, "All_Participants", y_col),
            }

    flattened = process.flatten_prediction_df(df)
    ne_options = {
        "extremes": [1, 99],
        "rounded": list(range(5, 100, 5)),
        "roundedAndExtremes": [1, 99] + list(range(5, 100, 5)),
    }
    ne_results = {}
    for key, ne in ne_options.items():
        chi_square_results, fractions_dist = chi_square_for_ne(flattened, ne)
        ne_results[key] = {
            "chiSquare": table_payload(chi_square_results),
            "bootstrap": bootstrap_histograms(fractions_dist),
        }

    payload = {
        "groupSizes": {key: int(mask.sum()) for key, mask in masks.items()},
        "segmentedHistograms": segmented_histograms,
        "metricsByQuestion": metrics_by_question,
        "scatter": scatter,
        "notEqualResults": ne_results,
    }
    write_json(DATA_DIR / "experience.json", payload)


def build_aggregation(df: pd.DataFrame, questions: list[dict], resolution_df: pd.DataFrame) -> None:
    ests = df.filter(like="@")
    original_predictions = ests.mean() / 100
    transformed_predictions = beta.cdf(original_predictions, 7, 7)

    masks = group_masks(df)
    ests_sf = ests.loc[masks["SF"]]
    weights = [0.05, 0.8, 0.1, 0.05]
    weighted_sum = (
        weights[0] * ests.mean(axis=0)
        + weights[1] * ests_sf.mean(axis=0)
        + weights[2] * ests.loc[masks["FE"]].mean(axis=0)
        + weights[3] * ests.loc[masks["LW"]].mean(axis=0)
    )
    weighted_transformed_mean = beta.ppf(weighted_sum / 100, 1 / 3, 1 / 3)
    my_preds = pd.DataFrame(np.round(weighted_transformed_mean * 100))
    my_preds[my_preds > 99] = 99
    my_preds[my_preds < 1] = 1

    prediction_rows = []
    question_lookup = {q["number"]: q for q in questions}
    for number, prediction in zip(resolution_df["question_number"], my_preds[0]):
        q_number = int(number)
        text = question_lookup[q_number]["text"]
        if ". " in text:
            text = text.split(". ", 1)[1]
        prediction_rows.append(
            {
                "Question": q_number,
                "Prediction": clean_scalar(prediction),
                "Outcome": question_lookup[q_number]["outcome"],
                "Question text": text,
            }
        )

    payload = {
        "betaScatter": {
            "x": clean_array(original_predictions, 6),
            "y": clean_array(transformed_predictions, 6),
        },
        "weightedScatter": {
            "x": clean_array(ests_sf.mean() / 100, 6),
            "y": clean_array(my_preds[0].to_numpy() / 100, 6),
        },
        "myPredictionsTable": {
            "columns": ["Question", "Prediction", "Outcome", "Question text"],
            "rows": [[row[col] for col in ["Question", "Prediction", "Outcome", "Question text"]] for row in prediction_rows],
        },
    }
    write_json(DATA_DIR / "aggregation.json", payload)


def step_distribution(my_scores, base_scores, bin_range, bins=50) -> dict:
    hist_base, edges_base = np.histogram(base_scores, bins=bins, range=bin_range, density=True)
    hist_my, edges_my = np.histogram(my_scores, bins=bins, range=bin_range, density=True)
    step_x_base = np.repeat(edges_base, 2)
    step_x_my = np.repeat(edges_my, 2)
    step_y_base = np.zeros(len(step_x_base))
    step_y_my = np.zeros(len(step_x_my))
    step_y_base[1:-1] = np.repeat(hist_base, 2)
    step_y_my[1:-1] = np.repeat(hist_my, 2)
    return {
        "base": {"x": clean_array(step_x_base, 6), "y": clean_array(step_y_base, 8)},
        "mine": {"x": clean_array(step_x_my, 6), "y": clean_array(step_y_my, 8)},
    }


def binary_sim_variant(my_predictions, sf_predictions, seed: int, base_preds_are_probs: bool, bin_range) -> dict:
    np.random.seed(seed)
    my_scores, base_scores, percentiles = util.sim_binary_comparison(
        1000, my_predictions, sf_predictions, base_preds_are_probs=base_preds_are_probs
    )
    return {
        "seed": seed,
        "scoreDistribution": step_distribution(my_scores, base_scores, bin_range),
        "percentileHistogram": histogram_payload(percentiles, bins=20, density=True),
        "medianPercentile": clean_scalar(np.median(percentiles)),
    }


def blind_mode_sim_variant(my_predictions, estimates_matrix, seed: int) -> dict:
    np.random.seed(seed)
    my_brier_score, blind_mode_scores, percentiles = util.sim_blind_mode_comparison(
        my_predictions, estimates_matrix
    )
    return {
        "seed": seed,
        "blindModeHistogram": histogram_payload(
            blind_mode_scores,
            bins=20,
            bin_range=(min(0.10, np.min(blind_mode_scores)), max(0.55, np.max(blind_mode_scores))),
        ),
        "myBrierScore": clean_scalar(my_brier_score),
        "percentileHistogram": histogram_payload(percentiles, bins=20),
        "meanPercentile": clean_scalar(np.mean(percentiles)),
        "finishPlace": util.ordinal(round(len(blind_mode_scores) * (np.mean(percentiles) / 100))),
        "winPercent": clean_scalar(np.mean(np.array(percentiles) < (100 / len(blind_mode_scores))) * 100),
        "participantCount": int(len(blind_mode_scores)),
    }


def build_simulations(df: pd.DataFrame) -> None:
    my_predictions, sf_predictions = load_predictions()
    estimates_matrix = process.get_estimates_matrix(df)
    seeds = [42, 4242, 9090, 1337, 86753, 12011, 55001, 73003]
    payload = {
        "superforecasterCalibration": [
            binary_sim_variant(my_predictions, sf_predictions, seed, True, (0.1, 0.35))
            for seed in seeds
        ],
        "perfectCalibration": [
            binary_sim_variant(my_predictions, sf_predictions, seed + 1, False, (0.075, 0.275))
            for seed in seeds
        ],
        "blindMode": [
            blind_mode_sim_variant(my_predictions, estimates_matrix, seed + 2)
            for seed in seeds
        ],
    }
    write_json(DATA_DIR / "simulations.json", payload)


def load_predictions() -> tuple[np.ndarray, np.ndarray]:
    sf = pd.read_csv(Path(process.RESULTS_FOLDER) / "sf_mean_predictions.csv").values.T[0] / 100
    sf = sf.round(2)
    mine = pd.read_csv(Path(process.RESULTS_FOLDER) / "my_final_predictions.csv").values.T[0] / 100
    return mine, sf


def build_markets(df: pd.DataFrame, questions: list[dict], resolution_df: pd.DataFrame) -> None:
    my_predictions, _ = load_predictions()
    market_hist_df = process.get_all_markets(silent=True)
    for q_number in range(1, 51):
        q_col = f"Q{q_number}"
        write_json(
            MARKET_DIR / f"q{q_number}.json",
            {
                "time": clean_array(market_hist_df["time"]),
                "probability": clean_array(market_hist_df[q_col], 6),
            },
        )

    my_pred_df = pd.DataFrame(columns=[f"Q{i}" for i in range(1, 51)])
    my_pred_df.loc[0] = my_predictions
    my_mse = process.get_score_df(my_pred_df, market_hist_df, score_type="mse")
    my_brier = process.get_score_df(my_pred_df, market_hist_df, score_type="brier")
    score_times = clean_array(my_mse["time"])
    score_payload = {
        "time": score_times,
        "mse": clean_array(my_mse[0], 8),
        "brier": clean_array(my_brier[0], 8),
    }

    estimates_matrix = process.get_estimates_matrix(df)
    resolution = resolution_df["resolution"].values
    blind_brier = np.mean(np.square(estimates_matrix - resolution), axis=1)
    sorted_brier = np.sort(blind_brier)
    aggregated_final_brier = np.mean(np.square(my_pred_df.values - resolution), axis=1)
    my_rank = np.searchsorted(sorted_brier, aggregated_final_brier)
    my_percentile = my_rank / len(sorted_brier)
    sf_briers = blind_brier[df["Superforecaster"] == "Yes"]
    sf_median = np.median(sf_briers)
    sf_median_rank = np.searchsorted(sorted_brier, sf_median)
    sf_median_percentile = sf_median_rank / len(sorted_brier)

    result_df = pd.DataFrame(
        {
            "Rank": np.searchsorted(sorted_brier, blind_brier) + 1,
            "Percentile": 100 - ((np.searchsorted(sorted_brier, blind_brier) + 1) / len(sorted_brier) * 100),
            "Brier score": blind_brier,
        }
    )
    for col in df.columns:
        if col.startswith("@"):
            result_df[col] = df[col]
    result_df = result_df.sort_values(by="Brier score")
    write_json(DATA_DIR / "market-final-table.json", table_payload(result_df))

    violin = {
        "all": clean_array(blind_brier, 8),
        "FE": clean_array(blind_brier[df["ForecastingExperience"] == "Yes"], 8),
        "LW": clean_array(blind_brier[df["LessWrong"] == "Yes"], 8),
        "SF": clean_array(sf_briers, 8),
    }
    payload = {
        "questions": questions,
        "myPredictions": clean_array(my_predictions, 6),
        "scoreTimeSeries": score_payload,
        "violin": violin,
        "summary": {
            "aggregatedFinalBrier": clean_scalar(aggregated_final_brier[0]),
            "topTenBrier": clean_array(sorted_brier[:10], 3),
            "myRank": util.ordinal(int(my_rank[0])),
            "myPercentile": clean_scalar(my_percentile[0]),
            "participantCount": int(len(sorted_brier)),
            "superforecasterMedian": clean_scalar(sf_median),
            "superforecasterRank": util.ordinal(int(sf_median_rank)),
            "superforecasterPercentile": clean_scalar(1 - sf_median_percentile),
        },
    }
    write_json(DATA_DIR / "markets.json", payload)


def get_aggregation_score_arguments(df: pd.DataFrame):
    weights_original = [0.05, 0.8, 0.1, 0.05]
    estimates_df = df.filter(like="@")
    mean_ests = np.array(estimates_df.mean(axis=0))
    mean_ests_sf = np.array(estimates_df.loc[df["Superforecaster"].values == "Yes"].mean(axis=0))
    mean_ests_fe = np.array(estimates_df.loc[df["ForecastingExperience"].values == "Yes"].mean(axis=0))
    mean_ests_lw = np.array(estimates_df.loc[df["LessWrong"].values == "Yes"].mean(axis=0))
    return weights_original, mean_ests, mean_ests_sf, mean_ests_fe, mean_ests_lw


def build_posthoc(df: pd.DataFrame, resolution_df: pd.DataFrame) -> None:
    resolution = resolution_df["resolution"].values
    weights_original, mean_ests, mean_ests_sf, mean_ests_fe, mean_ests_lw = get_aggregation_score_arguments(df)
    beta_range = np.logspace(-2, 2, 50)
    mesh_equal = util.generate_aggregate_meshgrid(*weights_original[1:], beta_range, equal_betas=True)
    score_equal = util.calculate_score_over_meshgrid(
        mesh_equal, mean_ests, mean_ests_sf, mean_ests_fe, mean_ests_lw, resolution
    )
    mesh_2d = util.generate_aggregate_meshgrid(*weights_original[1:], beta_range, equal_betas=False)
    score_2d = util.calculate_score_over_meshgrid(
        mesh_2d, mean_ests, mean_ests_sf, mean_ests_fe, mean_ests_lw, resolution
    )

    np.random.seed(0)
    weights_range = np.linspace(0, 1, 20)
    parameter_mesh = util.generate_aggregate_meshgrid(
        weights_range, weights_range, weights_range, [1 / 3], equal_betas=False, add_jitter=True
    )
    score_vec = util.calculate_score_over_meshgrid(
        parameter_mesh, mean_ests, mean_ests_sf, mean_ests_fe, mean_ests_lw, resolution
    )
    tsne_all = TSNE(n_components=2, random_state=0).fit_transform(parameter_mesh)
    tsne_minus_sf = TSNE(n_components=1, random_state=0).fit_transform(np.delete(parameter_mesh, 1, axis=1))
    tsne_minus_fe = TSNE(n_components=1, random_state=0).fit_transform(np.delete(parameter_mesh, 2, axis=1))

    payload = {
        "beta1d": {"x": clean_array(beta_range, 8), "y": clean_array(score_equal, 8)},
        "beta2d": {
            "x": clean_array(beta_range, 8),
            "y": clean_array(beta_range, 8),
            "z": [
                clean_array(row, 8)
                for row in score_2d.reshape((len(beta_range), len(beta_range)))
            ],
        },
        "weightScoreHistogram": histogram_payload(score_vec, bins=int(round(np.sqrt(len(score_vec))))),
        "tsne2d": {
            "x": clean_array(tsne_all[:, 0], 6),
            "y": clean_array(tsne_all[:, 1], 6),
            "score": clean_array(score_vec, 8),
        },
        "tsne1d": {
            "sfWeight": clean_array(parameter_mesh[:, 1], 6),
            "feWeight": clean_array(parameter_mesh[:, 2], 6),
            "minusSf": clean_array(tsne_minus_sf[:, 0], 6),
            "minusFe": clean_array(tsne_minus_fe[:, 0], 6),
            "score": clean_array(score_vec, 8),
        },
    }
    write_json(DATA_DIR / "posthoc.json", payload)


def feature_slug(feature_name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", feature_name).strip("-").lower()[:80]
    digest = hashlib.md5(feature_name.encode("utf-8")).hexdigest()[:8]
    return f"{base or 'feature'}-{digest}"


def build_supervised(df: pd.DataFrame, resolution_df: pd.DataFrame) -> None:
    feature_df = process.get_feature_df(df)
    estimates_matrix = process.get_estimates_matrix(df)
    resolution = resolution_df["resolution"].values
    brier_scores = np.mean(np.square(estimates_matrix - resolution), axis=1)
    r_and_p = util.correlate_features_to_score(feature_df, brier_scores)
    r_and_p = r_and_p.reset_index(names="feature")
    r_and_p["slug"] = r_and_p["feature"].map(feature_slug)
    r_and_p["effect"] = np.where(
        (r_and_p["p_value"] < 0.05) & (r_and_p["r_value"] < 0),
        "better",
        np.where((r_and_p["p_value"] < 0.05) & (r_and_p["r_value"] > 0), "worse", "not significant"),
    )
    r_and_p["neg_log10_p"] = -np.log10(r_and_p["p_value"])

    features = []
    for feature_name in sorted(feature_df.columns):
        slug = feature_slug(feature_name)
        x = feature_df[feature_name].to_numpy(dtype=float)
        y = brier_scores
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        unique_values = np.sort(np.unique(x))
        if len(unique_values) > 10:
            payload = {
                "feature": feature_name,
                "type": "continuous",
                "x": clean_array(x, 6),
                "y": clean_array(y, 8),
            }
        else:
            groups = []
            for value in unique_values:
                scores = y[x == value]
                if len(unique_values) == 2 and set(unique_values).issubset({0.0, 1.0}):
                    label = "Yes" if value == 1 else "No"
                else:
                    label = str(clean_scalar(value))
                groups.append(
                    {
                        "value": clean_scalar(value),
                        "label": label,
                        "scores": clean_array(scores, 8),
                    }
                )
            payload = {"feature": feature_name, "type": "discrete", "groups": groups}
        write_json(FEATURE_DIR / f"{slug}.json", payload)
        features.append({"name": feature_name, "slug": slug})

    payload = {
        "features": features,
        "defaultFeatureSlug": feature_slug("Income"),
        "correlations": records_payload(r_and_p.sort_values("r_value")),
        "volcano": records_payload(r_and_p),
    }
    write_json(DATA_DIR / "supervised.json", payload)


def main() -> None:
    reset_dist()
    df, _ = process.load_and_process_results()
    markets_df = process.get_current_probs(silent=True)
    resolution_df = process.get_target_df()
    questions = make_questions(markets_df, resolution_df)

    build_home(df, questions)
    build_experience(df)
    build_aggregation(df, questions, resolution_df)
    build_simulations(df)
    build_markets(df, questions, resolution_df)
    build_posthoc(df, resolution_df)
    build_supervised(df, resolution_df)


if __name__ == "__main__":
    main()
