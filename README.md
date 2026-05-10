# ACX Prediction Contest

Analysis of the Blind Mode predictions data from the Astral Codex 2023 prediction contest
to produce an answer for the Full Mode.

## Environment Setup (uv)

This project now uses `uv` for dependency and environment management.

```bash
uv sync --group dev --group notebooks
```

- Run tests:

```bash
uv run pytest -vv
```

## ACX links

- Announcement of the contest:
<https://www.astralcodexten.com/p/2023-prediction-contest>.
- Stage 2 of the contest, with the Blind Mode data:
<https://astralcodexten.substack.com/p/stage-2-of-prediction-contest>.
- Contest conclusion:
<https://www.astralcodexten.com/p/who-predicted-2023>.
- Published final scores:
<https://www.astralcodexten.com/p/open-thread-323>
(The final sores were initially published here
<https://www.astralcodexten.com/p/open-thread-322> but those turned out to be incorrect).

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

## Data

- `/raw/`: Raw data from the contest, should not be modified
- `/processed/`: results of analysis that are not tracked in git
- `/results/`: important results of analysis that are tracked in git

## Streamlit

Run the Streamlit app locally with

```bash
uv run streamlit run streamlit/Home.py
```

The Streamlit app ```./streamlit/Home.py``` is deployed at
<https://acx-prediction-contest.streamlit.app/>.

## Static app

This repo also includes a generated static version of the app for GitHub Pages.
Build it with:

```bash
uv run python scripts/build_static.py
```

The command writes a deployable multi-file site to `dist/`.
The static app keeps heavy calculations in the Python build step and renders
navigation, controls, charts, and tables in the browser.
The `Static Pages` GitHub Actions workflow builds `dist/` and deploys it with
GitHub Pages; for this repository, the project Pages URL is expected to be
<https://jbreffle.github.io/acx-prediction-contest/> once Pages is configured
to use GitHub Actions.

## src

- Functions and constants used across the project

## Manifold

Note: only the 1000 most recent bets for a market can be retrieved through the API.
See <https://docs.manifold.markets/api#get-v0bets>:

> limit: Optional. How many bets to return. The default and maximum are both 1000.

## Scoring

The scoring method for the contest was not specified when the contest was announced.
I used the Brier score for my analyses,
but at the conclusion of the contest it was announced that the
[Metaculus scoring function](<https://www.metaculus.com/help/scores-faq/>)
would be used.
I added additional analyses comparing how these two scoring methods compare.

The log score underpins all Metaculus scores, which is:
$$
\text{log score} = \log(P(\text{outcome}))
$$
where $P(\text{outcome})$ is the probability assigned to the outcome by the prediction.
Higher scores are better.
The log score is always negative,
and they say this has proved unintuitive for users,
so they have constructed the
[baseline score](<https://www.metaculus.com/help/scores-faq/#baseline-score>)
and the
[peer score](<https://www.metaculus.com/help/scores-faq/#peer-score>)
as more intuitive alternatives.

The general form of the baseline score is:
$$
\text{baseline score} = 100 \times \frac{\text{log score(prediction)} - \text{log score(baseline)}}{\text{scale}}
$$
where $baseline$ is the baseline prediction that weights all outcomes equally
and $scale$ is set so that a perfect prediction gives a score of $100$.

The general form of the peer score is:
$$
\text{peer score} = 100 \times \frac{1}{N} \sum_{i=1}^N \text{log score}(p) - \text{log score}(p_i)
$$
where $p$ is the scored prediction,
$N$ is the number of other predictions, and
$p_i$ is the $i$th other prediction.

## TODO

- Finish impementing the Metaculus scoring method
- Transfer supervised aggregation results to streamlit
- Optimize streamlit with plot and data caching
- Finish hyperopt hyperparameter optimization
- Modify all src.plot functions to modify and return an axis, rather than create and return a figure
- Does the Response rate by participant distribution match the beta distribution? Maybe with the exclusion of the excess of 100% response rates?
