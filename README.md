# ACX Prediction Contest

Analysis of the Blind Mode predictions data from the Astral Codex 2023 prediction contest
to produce an answer for the Full Mode.

Announcement of the contest:
<https://www.astralcodexten.com/p/2023-prediction-contest>

Stage 2 of the contest, with the Blind Mode data.
<https://astralcodexten.substack.com/p/stage-2-of-prediction-contest>

## Data

- raw: Raw data from the contest, should not be modified
- processed: results of analysis that are not tracked in git
- results: important results of analysis that are tracked in git

## Notebooks

- 0_exploration: Basic visualization and statistics of the data
- 1_aggregation: Uses beta-transformed arithmetic mean predictions, weighted by predictor background to generate answers
  - Creates files my_final_predictions and sf_mean_predictions if save_to_csv = True
- 2_outcome-simulation: Monte Carlo simulation of outcomes
- 3_manifold: Use the Manifold Markets API to get prediction market data and evaluate predictions across time
- 4_post_hoc_aggregation: Find the optimal aggregation parameters given actual outcomes
- 5_post_hoc_features: Evaluate what survey features correlate with final performance in the Blind Mode
- 5_post_hoc_nn: Train a neural network to predict final performance in the Blind Mode and then use it to aggregate predictions
- 5_post_hoc_xgb: Train an XGBRegressor to predict final performance in the Blind Mode and then use it to aggregate predictions
- 6_hyperopt_nn: Use hyperopt to optimize the hyperparameters of a neural network to predict final performance in the Blind Mode and then use it to aggregate predictions
- 6_hyperopt_xgb: Use hyperopt to optimize the hyperparameters of an XGBRegressor to predict final performance in the Blind Mode and then use it to aggregate predictions

## Streamlit

Run streamlit locally with

```bash
python -m streamlit run ./streamlit/Home.py
```

Streamlit app ```./streamlit/Home.py``` is deployed at <https://jbreffle.github.io/acx-app>

## src

- Functions and constants used across the project

## Manifold

Note: only the 1000 most recent bets for a market can be retrieved through the API.
See <https://docs.manifold.markets/api#get-v0bets>:

> limit: Optional. How many bets to return. The default and maximum are both 1000.

## TODO

- Transfer supervised aggregation results to streamlit
- Optimize streamlit with plot and data caching
- Finish hyperopt hyperparameter optimization
- Modify all src.plot functions to modify and return an axis, rather than create and return a figure
- Does the Response rate by participant distribution match the beta distribution? Maybe with the exclusion of the excess of 100% response rates?
