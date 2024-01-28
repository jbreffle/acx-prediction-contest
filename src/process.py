"""Helper functions for the 2023 blind mode forecasting project.
"""

# Imports
import os
from io import StringIO
from pathlib import Path
import warnings

import numpy as np
import yaml
import pandas as pd
import pyprojroot.here
import wordninja
import manifoldpy
import pyarrow.parquet as pq


# Config
with open(pyprojroot.here("config.yaml"), "r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

DEFAULT_DATA_PATH = pyprojroot.here(config["DATA_PATH"])
DEFAULT_DATA_PATH_XLSX = pyprojroot.here(config["DATA_PATH_XLSX"])
DEFAULT_DATA_PATH_CSV = pyprojroot.here(config["DATA_PATH_CSV"])
PROCESSED_DATA_FOLDER = pyprojroot.here(config["PROCESSED_DATA_FOLDER"])
RESULTS_FOLDER = pyprojroot.here(config["RESULTS_FOLDER"])

# Create default folders if they don't exists
Path(PROCESSED_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)


# Functions
def get_estimates_matrix(df, nan_method="median"):
    estimates_df = df.filter(like="@", axis=1)
    estimates_matrix = estimates_df.values
    # Replace nans based on nan_method string
    match nan_method:
        case "median":
            estimates_matrix = np.nan_to_num(
                estimates_matrix, nan=estimates_df.median()
            )
        case "mean":
            estimates_matrix = np.nan_to_num(estimates_matrix, nan=estimates_df.mean())
        case _:
            raise ValueError(f"nan_method {nan_method} not recognized")
    estimates_matrix = estimates_matrix / 100
    return estimates_matrix


def get_target_df(df_format=True):
    """Returns the target variable for the ml model, which is final score."""
    markets_df = get_current_probs(silent=True)
    if df_format:
        y = pd.DataFrame()
        y["question_number"] = markets_df["question_number"]
        y["resolution"] = [1 if x == "YES" else 0 for x in markets_df["resolution"]]
    else:
        y = np.array([1 if x == "YES" else 0 for x in markets_df["resolution"]])
    return y


def get_feature_df(df):
    """Convert the dataframe to a form ready for ml."""
    # TODO: consider using df.assign()
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    # Cleaned ouput feature df
    feature_df = pd.DataFrame()
    # Convert numerically coded columns to numeric
    columns_to_numeric = [
        "Age",
        "Children",
        "IQ",
        "SATscoreverbalreading",
        "SATscoremath",
        "PoliticalSpectrum",
        "PoliticalInterest",
        "GlobalWarming",
        "Immigration",
        "MinimumWage",
        "Feminism",
        "Abortion",
        "HumanBiodiversity",
        "DonaldTrump",
        "Income",
        "MoodScale",
        "Anxiety_A",
        "Childhood",
        "LifeSatisfaction",
        "JobSatisfaction",
        "SocialSatisfaction",
        "RomanticSatisfaction",
        "Risks",
        "Trustworthy",
        "STEM",
        "HowmanytimeshaveyougottenCOVID",
        "HowmanymonthsagodidyougetCOVID",
        "IfyouhavelingeringfatigueproblemsfromCOVIDhowbadarethey",
        "YouthoughtyourcountryapossCOVIDresponsewas",
        "Shouldtherebeamaskmandateonflights",
        "RegardlessofhowyoufeelaboutitpoliticallyhowdidCOVIDlockdownaffec",
        "WhatisyourBMI",
        "Ifyouansweredyesabovewhatpercentofyourweightdidyoulose",
        "Ifyouansweredyesabovehowlonghasitbeeninyearssincethen",
        "Ifyouansweredyesabovewhatpercentoftheweighthaveyousincegainedbac",
        "DoyouhaveacidrefluxquotGERDquotquotheartburnquot",
        "Regardlessofyourdiethowmuchdoyoulikecarbohydrateslikebreadandpas",
        "Howlongisyouraveragedailycommuteonewayinminutes",
        "Doyoupreferthebigcityorthesuburbs",
        "Wheredoyoulivenow",
        "Regardlessofwhatkindofurbanenvironmentyoupersonallypreferwhatare",
        "Howlonginyearshaveyoulivedinyourcurrentneighborhood",
        "Howdoyouthinklifeinyourneighborhoodhaschangedsinceyoufirstmovedi",
        "WhatisthehighestdoseofLSDyouaposveevertakeninmicrograms",
        "Doyouhavemigraines",
        "Howwouldyourateyourepisodicmemoryieabilitytorememberspecificthin",
        "DoyouexperienceASMR",
        "IfyouaposvetriedSSRIsfordepressionhowwelldidtheywork",
        "Didyouatanytimehearvoicessayingquiteafewwordsorsentenceswhenther",
        "Havetherebeentimesthatyoufeltthatagroupofpeoplewasplottingtocaus",
        "IfyouansweredquotYesquotabovehowdoyoufeelaboutit",
        "IfyouansweredquotNoquotabovehowdoyoufeelaboutit",
        "IfyouansweredquotYesquotabovehowmanyyearsoldwereyouwhentheygotdi",
        "Doyouthinkyouaposrebetterorworsethantheaveragepersonatsavingmone",
        "Attheendofanaverageyearwhatpercentofyourincomewillyouhavesaved",
        "Pleaseguesstheanswerwithoutcheckinganyothersourceuntilyouaredone",
        "Howoftendoyouwatcholdmovies",
        "PredictionquestionWhatdoyouthinkisthepercentchancethatAIwilldest",
        "PredictionquestionWhatdoyouthinkoneBitcoinwillbeworthindollarsin",
        "Yourclosefriendsnotcountingcurrentandformerromanticpartnersare",
        "Wouldyouquotwireheadquotifyouhadtheoption",
        "Howmuchdoyoutrustthemainstreammedia",
        "Howgoodasenseofdirectionorientationandnavigationdoyouhave",
        "SupposeItoldyouthatyourfirstguessabouthowthedistancebetweenParis",
        "Ifyouaposvetriedpsychotherapyforanyissuehowwelldiditwork",
    ]
    for col in columns_to_numeric:
        feature_df[col] = pd.to_numeric(df[col], errors="coerce")
    # Convert diagnosis columns to rank
    diagnosis_columns = [
        "Depression",
        "Anxiety",
        "OCD",
        "Eatingdisorder",
        "PTSD",
        "Alcoholism",
        "Drugaddiction",
        "Borderline",
        "Bipolar",
        "Autism",
        "ADHD",
        "Schizophrenia",
    ]
    diagnosis_dict = {
        " ": np.nan,
        "I don't have this condition and neither does anyone in my family": 0,
        "I have family members (within two generations) with this condition": 1,
        "I think I might have this condition, although I have never been formally diagnosed": 2,  # pylint: disable=line-too-long
        "I have a formal diagnosis of this condition": 3,
    }
    for col in diagnosis_columns:
        feature_df[col] = df[col].map(diagnosis_dict)
    # Convert Yes/No columns to boolean
    yes_no_columns = [
        "ForecastingExperience",
        "Superforecaster",
        "LessWrong",
        "EducationComplete",
        "Subscriber",
        "AccordingtoyourownbestjudgmenthaveyoueverhadLongCOVID",
        "HaveyougottenanydosesofaCOVIDvaccine",
        "Doyouusuallywearafacemaskwhengoingouttostoresorrestaurants",
        "Haveyoueverlost10ofyourweightonpurposethroughdietandexercise",
        "Doyouownacar",
        "Doyouhaveavaliddriversaposlicense",
        "Doyouidentifyashavingquotmultiplepersonalitiesquot",
        "Doyouhaveatleastonebrother",
        "Doyouhaveatleastonesister",
        "Doyouhaveatleastonemaletwinortripletetc",
        "Doyouhaveatleastonefemaletwinortripletetc",
        "Haveyoucutoffanyfamilymembersoverpolitics",
        "Haveanyfamilymemberscutyouoffoverpolitics",
    ]
    for col in yes_no_columns:
        feature_df[col] = df[col].map({"Yes": 1, "No": 0})
    # Convert degree to rank
    # Note: Not sure how to handle MD/JD/PhD, but not many MDs or JDs
    degree_dict = {
        " ": np.nan,
        "None": 0,
        "High school": 1,
        "2 year degree": 2,
        "Bachelor's degree": 3,
        "Master's degree": 4,
        "MD": 5,
        "JD": 5,
        "PhD": 6,
    }
    feature_df["Degree"] = df["Degree"].map(degree_dict)
    # "Male"
    feature_df["Male"] = df["Sex"].apply(
        lambda x: 1 if x == "Male" else np.nan if pd.isnull(x) else 0
    )
    # "Atheist" from "ReligiousViews"
    feature_df["Atheist"] = df["ReligiousViews"].apply(
        lambda x: 1 if ("atheist" in str(x).lower()) else np.nan if pd.isnull(x) else 0
    )
    # "Monogomous" from "Relationshipstyle"
    feature_df["Monogomous"] = df["Relationshipstyle"].apply(
        lambda x: 1
        if ("monogamous" in str(x).lower())
        else np.nan
        if pd.isnull(x)
        else 0
    )
    # "USA" from "Country"
    feature_df["USA"] = df["Country"].apply(
        lambda x: 1
        if ("united states" in str(x).lower())
        else np.nan
        if pd.isnull(x)
        else 0
    )
    # "Heterosexual" from "SexualOrientation"
    feature_df["Heterosexual"] = df["SexualOrientation"].apply(
        lambda x: 1
        if ("heterosexual" in str(x).lower())
        else np.nan
        if pd.isnull(x)
        else 0
    )
    # Consequentialist from MoralViews
    feature_df["Consequentialist"] = df["MoralViews"].apply(
        lambda x: 1
        if ("consequentialism" in str(x).lower())
        else np.nan
        if pd.isnull(x)
        else 0
    )
    # Vegetarian, 0 if "No", 1 if "No, but", 2 if "Yes", nan if nan
    feature_df["Vegetarian"] = df["Vegetarian"].apply(
        lambda x: 0
        if ("no" in str(x).lower())
        else 1
        if ("no, but" in str(x).lower())
        else 2
        if ("yes" in str(x).lower())
        else np.nan
    )
    # Handedness, 0 if right, 1 if left nan otherwise
    feature_df["LeftHanded"] = df["Handedness"].apply(
        lambda x: 0
        if ("right" in str(x).lower())
        else 1
        if ("left" in str(x).lower())
        else np.nan
    )
    # "Burp" from "Doyouburp"
    # "Never or almost never", "Less than once a week, but sometimes",
    # "L"ess than once a day, but more than once a week", "Yes, basically every day"
    feature_df["Burp"] = df["Doyouburp"].apply(
        lambda x: 0
        if ("never" in str(x).lower())
        else 1
        if ("less than once a week" in str(x).lower())
        else 2
        if ("less than once a day" in str(x).lower())
        else 3
        if ("yes" in str(x).lower())
        else np.nan
    )
    # "OwnCrypto" from "Doyouholdcryptocurrency" 0 if no, 1 if yes, nan otherwise
    feature_df["OwnCrypto"] = df["Doyouholdcryptocurrency"].apply(
        lambda x: 0
        if ("no" in str(x).lower())
        else 1
        if ("yes" in str(x).lower())
        else np.nan
    )
    # left/right from "PoliticalAffiliation" (1 if left, 0 if right, nan if neither)
    feature_df["PoliticalAffiliation"] = df["PoliticalAffiliation"].apply(
        lambda x: -1
        if (
            "liberal" in str(x).lower()
            or "marxist" in str(x).lower()
            or "social democratic" in str(x).lower()
        )
        else 1
        if (
            "alt-right" in str(x).lower()
            or "conservative" in str(x).lower()
            or ("neoreactionary" in str(x).lower())
        )
        else np.nan
        if pd.isnull(x)
        else 0
    )
    # Moved left/right from "PoliticalMovement"
    feature_df["PoliticalMovement"] = df["PoliticalChange"].apply(
        lambda x: -1
        if ("left" in str(x).lower())
        else 1
        if ("right" in str(x).lower())
        else np.nan
        if pd.isnull(x)
        else 0
    )
    # Misc clean up
    feature_df.rename(columns={"WhatisyourBMI": "BMI"}, inplace=True)
    long_paris_str = "SupposeItoldyouthatyourfirstguessabouthowthedistancebetweenParis"
    feature_df.rename(
        columns={long_paris_str: "distance_to_paris_estiamte"},
        inplace=True,
    )
    # Make outliers nan
    feature_df.loc[feature_df["Income"] > 1e6, "Income"] = np.nan
    feature_df.loc[feature_df["Age"] >= 100, "Age"] = np.nan
    feature_df.loc[feature_df["IQ"] >= 200, "IQ"] = np.nan
    feature_df.loc[feature_df["BMI"] >= 100, "BMI"] = np.nan
    feature_df.loc[feature_df["BMI"] < 10, "BMI"] = np.nan
    # Attheendofanaverageyearwhatpercentofyourincomewillyouhavesaved
    feature_df.loc[
        feature_df["Attheendofanaverageyearwhatpercentofyourincomewillyouhavesaved"]
        > 110,
        "Attheendofanaverageyearwhatpercentofyourincomewillyouhavesaved",
    ] = np.nan
    feature_df.loc[
        feature_df["HowmanymonthsagodidyougetCOVID"] > 40,
        "HowmanymonthsagodidyougetCOVID",
    ] = np.nan
    feature_df.loc[
        feature_df["SATscoreverbalreading"] > 800, "SATscoreverbalreading"
    ] = np.nan
    feature_df.loc[
        feature_df["SATscoreverbalreading"] < 200, "SATscoreverbalreading"
    ] = np.nan
    feature_df.loc[feature_df["SATscoremath"] > 800, "SATscoremath"] = np.nan
    feature_df.loc[feature_df["SATscoremath"] < 200, "SATscoremath"] = np.nan
    feature_df.loc[
        feature_df["distance_to_paris_estiamte"] > 2 * 1e5, "distance_to_paris_estiamte"
    ] = np.nan
    return feature_df


def abs_diff_from_50(flattened_prediction_df):
    """Returns a dataframe with the absolute difference from 50% for each prediction."""
    abs_diff_flattened_df = pd.DataFrame(
        {
            "class": flattened_prediction_df["class"],
            "prediction": abs(50 - flattened_prediction_df["prediction"]),
        }
    )
    return abs_diff_flattened_df


def flatten_prediction_df(blind_mode_df):
    """Returns a dataframe with all predictions flattened"""
    ests_fe = blind_mode_df.loc[
        blind_mode_df["ForecastingExperience"].values == "Yes"
    ].filter(like="@", axis=1)
    ests_sf = blind_mode_df.loc[
        blind_mode_df["Superforecaster"].values == "Yes"
    ].filter(like="@", axis=1)
    ests_lw = blind_mode_df.loc[blind_mode_df["LessWrong"].values == "Yes"].filter(
        like="@", axis=1
    )
    x_all = np.sort(
        blind_mode_df.filter(like="@", axis=1).dropna().to_numpy().flatten()
    )
    x_fe = np.sort(ests_fe.dropna().to_numpy().flatten())
    x_sf = np.sort(ests_sf.dropna().to_numpy().flatten())
    x_lw = np.sort(ests_lw.dropna().to_numpy().flatten())
    # Create a dataframe with columns 'class' and 'prediction'
    df = pd.DataFrame(
        {
            "class": ["all"] * len(x_all)
            + ["FE"] * len(x_fe)
            + ["SF"] * len(x_sf)
            + ["LW"] * len(x_lw),
            "prediction": np.concatenate([x_all, x_fe, x_sf, x_lw]),
        }
    )
    return df


def load_and_process_results(
    results_file=None, use_csv=False, fix_bad_rows=True, remove_bad_rows=True
):
    """Loads the results file and separates out the estimates.

    This function was originally written to load the results from the csv file,
    but there are delimiter issues that cause some rows to be shifted over.

    Preferentially loads .parquett over .xlsx over .csv.
    Saves the results to .parquett if it doesn't exist.

    Args:
        results_file: filepath to the results csv file

    Returns:
        data: the full dataframe
        ests: the estimates dataframe
    """
    # If results_file is None, then use the default based on which file type is available
    if results_file is None and not use_csv:
        if os.path.isfile(DEFAULT_DATA_PATH):
            results_file = DEFAULT_DATA_PATH
        elif os.path.isfile(DEFAULT_DATA_PATH_XLSX):
            results_file = DEFAULT_DATA_PATH_XLSX
        elif os.path.isfile(DEFAULT_DATA_PATH_CSV):
            results_file = DEFAULT_DATA_PATH_CSV
        else:
            raise ValueError("No results file found")
    elif results_file is None and use_csv:
        results_file = DEFAULT_DATA_PATH_CSV
    # Get extension from  PosixPath object
    file_type = results_file.suffix
    match file_type:
        case ".parquet":
            data = pq.read_table(results_file).to_pandas()
        case ".xlsx":
            data = pd.read_excel(results_file)
        case ".csv":
            # Some of the answers weren't wrapped in quotes but had commas in them
            # These two fixes correct 654 rows, but there are still 34 bad rows
            if fix_bad_rows:
                target_string_1 = "Never, I only lurk"
                target_string_2 = "No,Some other reason"
                # Read the file and replace the string
                with open(results_file, "r", encoding="utf-8") as file:
                    file_content = file.read()
                    modified_content = file_content.replace(
                        target_string_1, f'"{target_string_1}"'
                    )
                    modified_content = modified_content.replace(
                        target_string_2, f'"{target_string_2}"'
                    )
                # Use StringIO to create a file-like object from the modified content
                modified_file = StringIO(modified_content)
            else:
                modified_file = results_file
            # Load the modified content into a pandas DataFrame
            data = pd.read_csv(modified_file)
            # If a row has a value equal to np.isnan() in column "Unnamed: 225"
            # then remove it from the dataset blind_mode_df because one of its
            # question answers was improperly delimited,
            # which caused the entire row to be shifted over by one column
            if remove_bad_rows:
                data = data[pd.isnull(data["Unnamed: 225"])]
            # Remove bad columns, Unnamed: *
            bad_columns = [col for col in data.columns if "Unnamed" in col]
            data.drop(columns=bad_columns, inplace=True)
            print("WARNING: Loading the xlsx file is preffered due to delimiter issues")
        case _:
            raise ValueError(f"File type {file_type} not recognized")
    # Save to parquet if it doesn't exist
    if file_type != ".parquet":
        parquet_file = DEFAULT_DATA_PATH
        if not os.path.isfile(parquet_file):
            data.to_parquet(parquet_file)
    # Make blank lines np.nan
    data = data.replace(r"^\s*$", np.nan, regex=True)
    # Convert the question columns to numeric
    question_columns = [col for col in data.columns if "@" in col]
    data[question_columns] = data[question_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    # Separate out the estimates
    ests = data.filter(like="@", axis=1)
    ests = ests.apply(pd.to_numeric, errors="coerce")
    return data, ests


def get_manifold_slugs(data_file=DEFAULT_DATA_PATH):
    """..."""
    # Load data
    _, ests = load_and_process_results(data_file)
    # Get the column names of the estimates
    q_names = ests.columns
    word_list = wordninja.split(q_names[10])
    # join with dashes, and convert to lowercase
    q_name = "-".join(word_list).lower()
    # For each element of q_names, split it into words, and join with dashes, and
    # limit to 35 characters to match the manifold url slug
    list_of_slugs = []
    for q_name in q_names:
        word_list = wordninja.split(q_name)
        # join with dashes, and convert to lowercase
        q_name = "-".join(word_list).lower()
        q_name = q_name[:35]
        if q_name[-1] == "-":
            q_name = q_name[:-1]
        list_of_slugs.append(q_name)
    # Validated overrides for manifold url slug
    list_of_slugs[0] = "1-will-vladimir-putin-be-president"  # putin
    list_of_slugs[9] = "10-will-china-launch-a-fullscale-in"  # fullscale
    list_of_slugs[10] = "11-will-any-new-country-join-nato-i"
    list_of_slugs[11] = "12-will-ali-khameini-cease-to-be-su"  # khameini
    list_of_slugs[14] = "15-at-the-end-of-2023-will-predicti"  # Changed wording
    list_of_slugs[15] = "16-at-the-end-of-2023-will-predicti"  # Changed wording
    list_of_slugs[16] = "17-at-the-end-of-2023-will-predicti"  # Changed wording
    list_of_slugs[17] = "18-at-the-end-of-2023-will-predicti"  # Changed wording
    list_of_slugs[20] = "21-will-donald-trump-make-at-least"  # atleast -> at least
    list_of_slugs[24] = "25-will-rishi-sunak-be-prime-minist"  # sunak
    list_of_slugs[26] = "27-will-elon-musk-remain-owner-of-t"  # elon
    list_of_slugs[27] = "28-will-twitters-net-income-be-high"  # not sure...
    list_of_slugs[28] = "29-will-twitters-average-monetizabl"  # not sure...
    list_of_slugs[29] = "30-will-us-cpi-inflation-for-2023-a"  # us cpi
    list_of_slugs[30] = "31-will-the-sp-500-index-go-up-over"  # ampersand
    list_of_slugs[31] = "32-will-the-sp-500-index-reach-a-ne"  # ampersand
    list_of_slugs[32] = "33-will-the-shanghai-index-of-chine"  # index of
    list_of_slugs[35] = "36-will-tether-depeg-in-2023"  # depeg
    list_of_slugs[37] = "38-will-any-faang-or-musk-company-a"  # faang
    list_of_slugs[38] = "39-will-openai-release-gpt4-in-2023"
    list_of_slugs[39] = "40-will-spacexs-starship-reach-orbi"
    list_of_slugs[40] = "41-will-an-image-model-win-scott-al"
    list_of_slugs[41] = "42-will-covid-kill-at-least-50-as-m"
    list_of_slugs[42] = "43-will-a-new-version-of-covid-be-s"
    list_of_slugs[46] = "47-will-a-successful-deepfake-attem"
    list_of_slugs[48] = "49-will-ai-win-a-programming-compet"
    list_of_slugs[49] = "50-will-someone-release-dalle-but-f"
    return list_of_slugs


def get_current_probs(current_prob_file=None, silent=False):
    """Get the current probabilities from manifold and save to csv.

    If the csv already exists, load it instead of fetching from manifold
    """
    if current_prob_file is None:
        current_prob_file = os.path.join(RESULTS_FOLDER, "current_manifold_prob.csv")
    if os.path.isfile(current_prob_file):
        # Load it
        current_manifold_df = pd.read_csv(current_prob_file)
        if not silent:
            print("Loading current probabilities from file")
    else:
        # Get market current market data for each element of list_of_slugs
        list_of_slugs = get_manifold_slugs()
        current_manifold_df = pd.DataFrame(list_of_slugs, columns=["slug"])
        current_manifold_df["market"] = current_manifold_df["slug"].apply(
            manifoldpy.api.get_slug
        )
        current_manifold_df["question_number"] = current_manifold_df["slug"].apply(
            lambda x: x.split("-")[0]
        )
        current_manifold_df["question"] = current_manifold_df["market"].apply(
            lambda x: x.question
        )
        current_manifold_df["probability"] = current_manifold_df["market"].apply(
            lambda x: x.probability
        )
        current_manifold_df["resolution"] = current_manifold_df["market"].apply(
            lambda x: x.resolution
        )
        # Save to csv
        current_manifold_df.drop(columns=["slug", "market"], inplace=True)
        current_manifold_df.to_csv(current_prob_file, index=False)
        if not silent:
            print("Fetching current probabilities")
    return current_manifold_df


def get_my_predictions(file_path=None):
    """Loads and returns my predictions from the csv file as a dataframe"""
    if file_path is None:
        file_path = os.path.join(RESULTS_FOLDER, "SUBMITED_PREDICTIONS.csv")
    my_predictions_df = pd.DataFrame()
    my_predictions_df["question_number"] = [(i + 1) for i in range(50)]
    my_predictions_df["my_prediction"] = pd.read_csv(file_path)["my_predictions"]
    my_predictions_df["probability"] = my_predictions_df["my_prediction"] / 100
    my_predictions_df["binary"] = my_predictions_df["probability"] > 0.5
    return my_predictions_df


def get_all_markets(market_hist_file=None, silent=False):
    """Get all markets from manifold.

    Load from csv if it exists, otherwise fetch from manifold with api.
    Save to csv after fetching if it doesn't exist.

    """
    if market_hist_file is None:
        market_hist_file = os.path.join(RESULTS_FOLDER, "all_market_hist.csv")
    if os.path.isfile(market_hist_file):
        # Load it
        market_hist_df = pd.read_csv(market_hist_file)
        if not silent:
            print("Loading market histories from file")
    else:
        # Get the correct list of manifold url slugs for the questions
        list_of_slugs = get_manifold_slugs()
        # For each new ith_slug, add the new times and probabilities to the dataframe
        # and merge the time vectors
        market_hist_df = pd.DataFrame()
        for ith_slug in range(len(list_of_slugs)):
            slug = list_of_slugs[ith_slug]
            slug_market = manifoldpy.api.get_slug(slug)
            market = manifoldpy.api.get_full_market(slug_market.id)
            times, probabilities = market.probability_history()
            market_df = pd.DataFrame(
                {"time": times, "Q" + str(ith_slug + 1): probabilities}
            )
            market_df.set_index("time", inplace=True)
            market_hist_df = market_hist_df.join(market_df, how="outer")
        market_hist_df = market_hist_df.interpolate(
            method="index", limit_direction="both"
        )
        market_hist_df.head()
        # Save to csv in ./results/
        market_hist_df.to_csv(market_hist_file)
        if not silent:
            print("Fetching market histories")
    return market_hist_df


def calc_score_series(predictions, market_hist_df, score_type="mse"):
    """Calculates the MSE or Brier score of a set of predictions across time

    Args:
        predictions: a pandas dataframe of predictions, with columns Q1 to Q50.
            Can be either a single set of predictions, or a set of predictions for each
            participant.
        market_hist_df: a pandas dataframe of market histories. There is a time column
            and a column for each question Q1 to Q50.
        score_type: Determines which score is calculated and returned,
            either 'mse' or 'brier'.
        verify_brier_format: if True and score_type is 'brier', then verify that
            market_hist_df is binary.

    Returns:
        score_df: a pandas dataframe of scores, with columns Q1 to Q50
    """
    if score_type not in ["mse", "brier"]:
        raise ValueError("score_type must be either 'mse' or 'brier'")
    # If score_type is brier, then convert market_hist_df to binary
    if score_type == "brier":
        # The most-likely outcome across time for each market
        # At each moment in time for each question, is the probability of the market
        # greater than 0.5?
        market_hist_df = market_hist_df > 0.5
    # For each question
    market_hist_df = market_hist_df.drop(columns=["time"])
    prediction_values = predictions.iloc[0, :]
    score_df = (prediction_values - market_hist_df) ** 2
    return score_df.mean(axis=1)


def get_score_df(
    predictions,
    market_hist_df,
    score_type="mse",
    load_from_file=True,
    save_to_file=True,
    score_df_file=None,
    use_subsample=True,
):
    """Returns a dataframe of the score across time for each participant.

    Creates a function that calls calc_score_series() for each participant and returns a
    dataframe of the results. Rows are time points, columns are participants.

    If use_subsample, then calculate the score for every 10th time point.
    """
    if score_df_file is None:
        if use_subsample:
            tmp_str = "_subsample"
        else:
            tmp_str = ""
        score_filename = f"blind_mode_scores_{score_type}{tmp_str}.csv"
        score_df_file = os.path.join(PROCESSED_DATA_FOLDER, score_filename)
    is_large_file = len(predictions) > 100
    if load_from_file and os.path.isfile(score_df_file) and is_large_file:
        # Load and return saved  scores
        score_df = pd.read_csv(score_df_file)
        # score_df has "time" as the first column
        # Convert all columns of score_df except "time" to numeric
        column_name_strs = score_df.columns[1:]
        column_name_ints = [int(x) for x in column_name_strs]
        column_name_dict = dict(zip(column_name_strs, column_name_ints))
        score_df.rename(columns=column_name_dict, inplace=True)
        return score_df
    # Calculate the score for each participant (each row of predictions)
    subsample_rate = 10
    if use_subsample:
        # Use the subsample rate, but also include the first and last time points
        subsample_times = market_hist_df["time"][::subsample_rate].values
        subsample_times = np.append(subsample_times, market_hist_df["time"].values[-1])
        subsample_times = np.insert(
            subsample_times, 0, market_hist_df["time"].values[0]
        )
        market_hist_df = market_hist_df.loc[
            market_hist_df["time"].isin(subsample_times)
        ]
    n_rows = len(market_hist_df)
    n_columns = len(predictions)
    tmp_matrix = np.zeros((n_rows, n_columns))
    for ith_participant in range(len(predictions)):
        participant_series = calc_score_series(
            predictions.iloc[[ith_participant]], market_hist_df, score_type=score_type
        )
        tmp_matrix[:, ith_participant] = participant_series
    output = pd.DataFrame(tmp_matrix)
    output.insert(0, "time", market_hist_df["time"].values)
    # If save_to_file, then save to file
    if save_to_file and is_large_file:
        output.to_csv(score_df_file, index=False)
    return output
