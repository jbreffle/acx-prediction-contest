# ACX Prediction Contest

Analysis of the Blind Mode predictions data to produce an answer for the Full Mode.

Announcement of the contest:
<https://www.astralcodexten.com/p/2023-prediction-contest>

Stage 2 of the contest, with the Blind Mode data.
<https://astralcodexten.substack.com/p/stage-2-of-prediction-contest>

## Data

- raw: Raw data from the contest, should not be modified
- processed: results of analysis that are not tracked in git
- results: important results of analysis that are tracked in git

## Notebooks

- exploration: Basic visualization and statistics of the data
- aggregation: Uses beta-transformed arithmetic mean predictions, weighted by predictor background to generate answers
  - Creates files my_final_predictions and sf_mean_predictions if save_to_csv = True
- D3-export: Computes data for the D3 visualization on jbreffle.github.io, saved as csv files
- manifold: Test the use of the Manifold API to get prediction market data
- outcome-simulation: Monte Carlo simulation of outcomes
- sns-exploration: Some tests of using sns for plotting

## Streamlit

Run streamlit locally with

```bash
python -m streamlit run ./streamlit/Home.py
```

Streamlit app ```./streamlit/Home.py``` is deployed at <https://jbreffle.github.io/acx-app>
Streamlit cheatsheet: <https://cheat-sheet.streamlit.app/>

## src

- Functions and constants used across the project

## Manifold

Note: only the 1000 most recent bets for a market can be retrieved through the API. See <https://docs.manifold.markets/api#get-v0bets>

> limit: Optional. How many bets to return. The default and maximum are both 1000.

## TODO

- Update README.md
- Finish remaining analysis in the post_hoc notebooks and then transfer to streamlit
- Modify all src.plot functions to modify and return an axis, rather than create and return a figure
- Use iframes to embed the streamlit app in a page on jbreffle.github.io
  - e.g. <https://elc.github.io/posts/streamlit-google-analytics/>
  - Make sure GA is working
  
Possibilities:

- Switch from using scipy.stats.beta.ppf to using scipy.stats.beta.cdf
- Does the Response rate by participant distribution match the beta distribution? Maybe with the exclusion of the excess of 100% response rates?
