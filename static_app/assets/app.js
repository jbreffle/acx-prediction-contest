const routes = [
  ["home", "Home"],
  ["experience", "Predictions by experience"],
  ["aggregation", "Aggregating predictions"],
  ["simulations", "Simulating outcomes"],
  ["markets", "Prediction markets"],
  ["posthoc", "Post-hoc aggregation"],
  ["supervised", "Supervised aggregation"],
];

const cache = new Map();
const pageState = {
  simSf: 0,
  simPerfect: 0,
  simBlind: 0,
  marketQuestion: 46,
  marketPrediction: null,
  marketAllPoints: false,
  featureSlug: null,
};

const groupLabels = {
  all: "All participants",
  FE: "Forecasting experience",
  LW: "LessWrong member",
  SF: "Super forecaster",
};

const groupColors = {
  all: "#2d3430",
  FE: "#d8a31d",
  LW: "#c87929",
  SF: "#b64242",
  better: "#257a45",
  worse: "#b64242",
  "not significant": "#8b948f",
};

const plotConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function getJson(path) {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(path).then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load ${path}: ${response.status}`);
        }
        return response.json();
      }),
    );
  }
  return cache.get(path);
}

function renderNav(active) {
  $("nav").innerHTML = routes
    .map(
      ([key, label]) =>
        `<a href="#${key}" class="${key === active ? "active" : ""}">${label}</a>`,
    )
    .join("");
}

function pageTitle(title, copy = "") {
  return `<section class="section"><h1 class="page-title">${title}</h1>${copy}</section>`;
}

function section(title, content) {
  return `<section class="section"><h2>${title}</h2>${content}</section>`;
}

function chartDiv(id, extraClass = "") {
  return `<div id="${id}" class="chart ${extraClass}"></div>`;
}

function baseLayout(title, overrides = {}) {
  return {
    title: title ? { text: title, x: 0.02, font: { size: 16 } } : undefined,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: "#19201d" },
    margin: { l: 62, r: 24, t: title ? 54 : 28, b: 58 },
    hovermode: "closest",
    legend: { orientation: "h", y: 1.12 },
    ...overrides,
  };
}

function plot(id, traces, layout = {}) {
  Plotly.react(id, traces, baseLayout(layout.title, layout), plotConfig);
}

function fmtPercent(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return "N/A";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtNumber(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toFixed(digits);
}

function questionByNumber(questions, number) {
  return questions.find((question) => question.number === Number(number));
}

function tableShell(id, note = "") {
  return `
    ${note ? `<p class="note">${note}</p>` : ""}
    <div id="${id}"></div>
  `;
}

function renderTable(id, payload, options = {}) {
  const root = $(id);
  const columns = payload.columns || [];
  let rows = payload.rows || [];
  const pageSize = options.pageSize || 25;
  const longColumns = new Set(options.longColumns || []);
  const state = { page: 0, sortIndex: null, sortDirection: 1, query: "" };

  root.innerHTML = `
    <div class="table-tools">
      <input type="search" placeholder="Search table" aria-label="Search table" />
      <span class="table-count"></span>
    </div>
    <div class="table-wrap"><table><thead></thead><tbody></tbody></table></div>
    <div class="pager">
      <button class="secondary prev">Previous</button>
      <span class="page-label"></span>
      <button class="secondary next">Next</button>
    </div>
  `;

  const search = root.querySelector("input");
  const thead = root.querySelector("thead");
  const tbody = root.querySelector("tbody");
  const count = root.querySelector(".table-count");
  const pageLabel = root.querySelector(".page-label");
  const prev = root.querySelector(".prev");
  const next = root.querySelector(".next");

  function filteredRows() {
    let output = rows;
    if (state.query) {
      output = output.filter((row) =>
        row.some((cell) => String(cell ?? "").toLowerCase().includes(state.query)),
      );
    }
    if (state.sortIndex != null) {
      output = [...output].sort((a, b) => {
        const av = a[state.sortIndex];
        const bv = b[state.sortIndex];
        const an = Number(av);
        const bn = Number(bv);
        let compare;
        if (!Number.isNaN(an) && !Number.isNaN(bn)) {
          compare = an - bn;
        } else {
          compare = String(av ?? "").localeCompare(String(bv ?? ""));
        }
        return compare * state.sortDirection;
      });
    }
    return output;
  }

  function draw() {
    const filtered = filteredRows();
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    state.page = Math.min(state.page, pages - 1);
    const start = state.page * pageSize;
    const visible = filtered.slice(start, start + pageSize);
    thead.innerHTML = `<tr>${columns
      .map((column, index) => `<th data-index="${index}">${escapeHtml(column)}</th>`)
      .join("")}</tr>`;
    tbody.innerHTML = visible
      .map(
        (row) =>
          `<tr>${row
            .map((cell, index) => {
              const cls = longColumns.has(columns[index]) ? "long" : "";
              return `<td class="${cls}">${escapeHtml(cell)}</td>`;
            })
            .join("")}</tr>`,
      )
      .join("");
    count.textContent = `${filtered.length.toLocaleString()} rows`;
    pageLabel.textContent = `Page ${state.page + 1} of ${pages}`;
    prev.disabled = state.page === 0;
    next.disabled = state.page >= pages - 1;
  }

  search.addEventListener("input", () => {
    state.query = search.value.trim().toLowerCase();
    state.page = 0;
    draw();
  });
  prev.addEventListener("click", () => {
    state.page -= 1;
    draw();
  });
  next.addEventListener("click", () => {
    state.page += 1;
    draw();
  });
  thead.addEventListener("click", (event) => {
    const th = event.target.closest("th");
    if (!th) return;
    const index = Number(th.dataset.index);
    if (state.sortIndex === index) {
      state.sortDirection *= -1;
    } else {
      state.sortIndex = index;
      state.sortDirection = 1;
    }
    draw();
  });
  draw();
}

function metricCards(metrics) {
  const all = metrics.all;
  return `
    <div class="metric-grid">
      <div class="metric"><span class="label">All participants</span><span class="value">${fmtNumber(all, 2)}</span></div>
      <div class="metric"><span class="label">Forecasting experience</span><span class="value">${fmtNumber(metrics.FE, 2)}</span><span class="delta">${fmtNumber(metrics.FE - all, 2)} vs all</span></div>
      <div class="metric"><span class="label">LessWrong member</span><span class="value">${fmtNumber(metrics.LW, 2)}</span><span class="delta">${fmtNumber(metrics.LW - all, 2)} vs all</span></div>
      <div class="metric"><span class="label">Super forecaster</span><span class="value">${fmtNumber(metrics.SF, 2)}</span><span class="delta">${fmtNumber(metrics.SF - all, 2)} vs all</span></div>
    </div>
  `;
}

function plotHistogram(id, payload, title, axis = {}) {
  plot(
    id,
    [
      {
        type: "bar",
        x: payload.x,
        y: payload.y,
        marker: { color: "#0b7261" },
        hovertemplate: "%{x}<br>%{y}<extra></extra>",
      },
    ],
    {
      title,
      xaxis: { title: axis.x || "" },
      yaxis: { title: axis.y || "", tickformat: axis.percent ? ".0%" : undefined },
    },
  );
}

function plotSegmentedHistogram(id, payload, title) {
  const traces = Object.entries(payload).map(([key, hist]) => ({
    type: "scatter",
    mode: "lines+markers",
    name: groupLabels[key],
    x: hist.x,
    y: hist.y,
    line: { color: groupColors[key], width: 2 },
  }));
  plot(id, traces, {
    title,
    xaxis: { title: "Predicted probability (binned)" },
    yaxis: { title: "% of predictions", tickformat: ".0%" },
  });
}

async function renderHome() {
  const data = await getJson("./data/home.json");
  $("app").innerHTML =
    pageTitle(
      "ACX 2023 Prediction Contest",
      `<div class="lead">
        <p>This app explores predictions made by participants in the
        <a href="https://www.astralcodexten.com/p/2023-prediction-contest" target="_blank" rel="noreferrer">ACX 2023 Prediction Contest</a>.</p>
        <p>The contest included both a <a href="https://www.astralcodexten.com/p/2023-prediction-contest" target="_blank" rel="noreferrer">Blind Mode</a>
        and a <a href="https://www.astralcodexten.com/p/stage-2-of-prediction-contest" target="_blank" rel="noreferrer">Full Mode</a>.
        This app lets you explore the Blind Mode participant predictions and shows the aggregation approach used to generate Full Mode predictions.</p>
        <p>The original Streamlit app is hosted at
        <a href="https://acx-prediction-contest.streamlit.app/" target="_blank" rel="noreferrer">acx-prediction-contest.streamlit.app</a>,
        and the project source is at
        <a href="https://github.com/jbreffle/acx-prediction-contest" target="_blank" rel="noreferrer">github.com/jbreffle/acx-prediction-contest</a>.</p>
      </div>`,
    ) +
    section(
      "App pages",
      `<p>This app has multiple pages, each of which address different questions:</p>
       <ul>
        <li><strong>Home</strong>: What are some basic properties of the Blind Mode predictions?</li>
        <li><strong>Predictions by experience</strong>: Does self-reported forecasting experience correlate with any property of the Blind Mode predictions?</li>
        <li><strong>Aggregating predictions</strong>: Can we aggregate Blind Mode predictions to generate more accurate predictions?</li>
        <li><strong>Simulating outcomes</strong>: How do our aggregated predictions fare in Monte Carlo simulations of possible futures?</li>
        <li><strong>Prediction markets</strong>: How well do our aggregate predictions compare to prediction markets across the year?</li>
        <li><strong>Post-hoc aggregation</strong>: How close to optimal were our aggregation parameters?</li>
        <li><strong>Supervised aggregation</strong>: What attributes of Blind Mode participants are most predictive of accurate forecasting?</li>
       </ul>
       <p>Note: The contest has now concluded and the results were announced
       <a href="https://www.astralcodexten.com/p/who-predicted-2023" target="_blank" rel="noreferrer">here</a>.</p>`,
    ) +
    section(
      "Blind Mode raw data",
      `<p>The full dataset of the Blind Mode contest includes each participant's predictions for each of the 50 questions and their self-reported forecasting experience. The data also includes responses to a large set of survey questions for the subset of participants who chose to complete them.</p>
       <details>
        <summary>View raw data</summary>
        <div class="details-body">
          <p class="note">The full processed table is loaded only when requested because it has 3,295 rows and 225 columns.</p>
          <button id="load-raw" class="secondary">Load raw data table</button>
          <div id="raw-table"></div>
        </div>
       </details>`,
    ) +
    section(
      "Prediction distributions",
      `<p>Use the slider to select one of the 50 questions of the contest. The histogram shows the distribution of all Blind Mode participant predictions for that question. Since the contest is over, the outcome is included.</p>
       <div class="control-row">
         <div class="control"><label for="home-question">Select question <output id="home-question-output">1</output></label><input id="home-question" type="range" min="1" max="50" value="1" /></div>
       </div>
       <p id="home-question-text"></p>
       ${chartDiv("home-question-chart")}`,
    ) +
    section(
      "Page links",
      `<p>Use the links below or the sidebar navigation.</p>
       <div class="card-grid">
        ${routes
          .filter(([key]) => key !== "home")
          .map(([key, label]) => `<a class="page-card" href="#${key}"><h3>${label}</h3><p>${pageCardText(key)}</p></a>`)
          .join("")}
       </div>`,
    ) +
    section(
      "Appendix: Response rates",
      `<p>Participants did not have to answer all questions. If a question was left blank, the participant received the average score for that question.</p>
       ${chartDiv("response-question-chart")}
       <p>Most questions had a high resposne rate. Two questions are slight outliers, with low response rates. They are:</p>
       <ul>
        <li>Question 36. Will Tether de-peg in 2023? Which had a response rate of 75.99%</li>
        <li>Question 41. Will an image model win Scott Alexander's bet on compositionality, to Edwin Chen's satisfaction, in 2023? Which had a response rate of 76.93%</li>
       </ul>
       <p>This result makes sense, as both of these questions are two of the most obscure and niche-topic questions in the contest.</p>
       ${chartDiv("response-participant-chart")}
       <p>Most participants answered most questions, many participants answered all questions, and a few participants answered zero questions.</p>
       <ul>
        <li>${data.responseSummary.allQuestionsPercent}% of participants answered all questions.</li>
        <li>${data.responseSummary.zeroQuestionsCount} participants answered zero questions.</li>
       </ul>`,
    );

  function updateQuestion() {
    const number = Number($("home-question").value);
    $("home-question-output").textContent = number;
    const question = questionByNumber(data.questions, number);
    $("home-question-text").innerHTML = `Question: ${escapeHtml(question.text)}<br />Outcome: <span class="outcome ${question.outcome.toLowerCase()}">${question.outcome}</span>`;
    plotHistogram("home-question-chart", data.predictionHistograms[number], "Prediction distribution", {
      x: "Predicted probability",
      y: `% of predictions (n=${data.predictionHistograms[number].n})`,
      percent: true,
    });
  }
  $("home-question").addEventListener("input", updateQuestion);
  $("load-raw").addEventListener("click", async () => {
    $("raw-table").innerHTML = '<p class="loading">Loading raw data...</p>';
    const raw = await getJson("./data/raw-data.json");
    renderTable("raw-table", raw, { pageSize: 20 });
    $("load-raw").remove();
  });
  updateQuestion();
  plotHistogram("response-question-chart", data.responseByQuestion, "Response rate by question", {
    x: "Fraction of participants answering the question",
    y: "Number of questions",
  });
  plotHistogram("response-participant-chart", data.responseByParticipant, "Response rate by participant", {
    x: "Fraction of questions predicted by participant",
    y: "Number of participants",
  });
}

function pageCardText(key) {
  return {
    experience: "Does self-reported forecasting experience correlate with predictions?",
    aggregation: "Can we aggregate Blind Mode predictions to generate more accurate predictions?",
    simulations: "How do our aggregate predictions fare in Monte Carlo simulations of possible futures?",
    markets: "How well do our aggregate predictions compare to prediction markets across the year?",
    posthoc: "What aggregation parameters would have performed best after outcomes were known?",
    supervised: "What attributes of Blind Mode participants are most predictive of accurate forecasting?",
  }[key];
}

async function renderExperience() {
  const [home, data] = await Promise.all([
    getJson("./data/home.json"),
    getJson("./data/experience.json"),
  ]);
  $("app").innerHTML =
    pageTitle(
      "Does self-reported forecasting experience correlate with predictions?",
      `<p class="lead">All Blind Mode participants self-identified among several categories of forecasting experience. There were ${data.groupSizes.all} total participaints. ${data.groupSizes.FE} participants reported having forecasting experience, ${data.groupSizes.LW} participants reported being members of LessWrong, and ${data.groupSizes.SF} participants reported being super forecasters.</p>
       <p>This page examines whether there are any differences in the predictions made by these groups. For analyses evaluating the final performance of these groups, see the Prediction markets and Supervised aggregation pages.</p>`,
    ) +
    section(
      "Prediction distributions by group",
      `<p>Use the slider to select a question. The histogram shows the distribution of predictions split by self-identified groups.</p>
       <div class="control-row"><div class="control"><label for="experience-question">Select question <output id="experience-question-output">1</output></label><input id="experience-question" type="range" min="1" max="50" value="1" /></div></div>
       <p id="experience-question-text"></p>
       ${chartDiv("experience-question-chart", "tall")}
       <p>Question mean by group:</p>
       <div id="experience-metrics"></div>
       <p>The mean prediction for each question is similar across groups. The difference of any group mean relative to the overal mean is generally not more than a few percentage points.</p>`,
    ) +
    section(
      "Average predictions are similar across self-identified groups",
      `<p>Rather than examining each question individually, we can ask if there are systematic differences in the average predictions across groups for the entire set of questions. Each point below is a question; the x value is the average prediction for all participants and the y value is the average prediction for a group.</p>
       <h3>Median predictions for each question split by group: all vs {FE, LW, SF}</h3>
       <div class="chart-grid">${chartDiv("scatter-median-FE", "small")}${chartDiv("scatter-median-LW", "small")}${chartDiv("scatter-median-SF", "small")}</div>
       <h3>Mean predictions for each question split by group: all vs {FE, LW, SF}</h3>
       <div class="chart-grid">${chartDiv("scatter-mean-FE", "small")}${chartDiv("scatter-mean-LW", "small")}${chartDiv("scatter-mean-SF", "small")}</div>
       <p>The average predictions for each group are quite similar to the overal average.</p>`,
    ) +
    section(
      "Rounded and extreme predictions",
      `<p>Although the average predictions are similar across groups, there might be differences in how participants make predictions. In particular, forecasting experience might affect whether participants make extreme predictions (1% or 99%) or round their predictions by 5.</p>
       <h3>The LessWrong and Forecasting Experience groups are more likely to avoid predicting 1% or 99%, but not the Superforecasters group</h3>
       ${tableShell("ne-extremes-table")}
       ${chartDiv("ne-extremes-chart")}
       <h3>Those with forecasting experience are less likely to round their predictions by 5s.</h3>
       ${tableShell("ne-rounded-table")}
       ${chartDiv("ne-rounded-chart")}
       <h3>All groups are more likely to avoid both rounding and extremes.</h3>
       ${tableShell("ne-rounded-extremes-table")}
       ${chartDiv("ne-rounded-extremes-chart")}`,
    );

  function updateQuestion() {
    const number = Number($("experience-question").value);
    $("experience-question-output").textContent = number;
    const question = questionByNumber(home.questions, number);
    $("experience-question-text").innerHTML = `Question: ${escapeHtml(question.text)}<br />Outcome: <span class="outcome ${question.outcome.toLowerCase()}">${question.outcome}</span>`;
    plotSegmentedHistogram("experience-question-chart", data.segmentedHistograms[number], "Prediction distributions by group");
    $("experience-metrics").innerHTML = metricCards(data.metricsByQuestion[number]);
  }

  function plotScatter(stat, group) {
    const item = data.scatter[stat][group];
    const x = item.points.map((point) => point.x);
    const y = item.points.map((point) => point.y);
    plot(
      `scatter-${stat}-${group}`,
      [
        { type: "scatter", mode: "markers", name: groupLabels[group], x, y, marker: { color: groupColors[group], size: 7 } },
        { type: "scatter", mode: "lines", name: "Equality", x: [0, 100], y: [0, 100], line: { color: "#8b948f", dash: "dash" } },
        { type: "scatter", mode: "lines", name: "Regression", x: item.regression.line.x, y: item.regression.line.y, line: { color: "#b64242" } },
      ],
      {
        title: groupLabels[group],
        xaxis: { title: "All participants", range: [0, 100] },
        yaxis: { title: groupLabels[group], range: [0, 100] },
        annotations: [
          {
            x: 5,
            y: 92,
            xref: "x",
            yref: "y",
            text: `p = ${fmtNumber(item.regression.p, 3)}<br>angle = ${fmtNumber(item.regression.angle, 1)} deg`,
            showarrow: false,
            align: "left",
            font: { color: "#b64242" },
          },
        ],
      },
    );
  }

  function plotNe(key, tableId, chartId, title) {
    const item = data.notEqualResults[key];
    renderTable(tableId, item.chiSquare, { pageSize: 10 });
    const traces = Object.entries(item.bootstrap).map(([group, hist]) => ({
      type: "bar",
      name: groupLabels[group],
      x: hist.x,
      y: hist.y,
      opacity: 0.62,
      marker: { color: groupColors[group] },
    }));
    plot(chartId, traces, {
      title,
      barmode: "overlay",
      xaxis: { title },
      yaxis: { title: "Count (bootstrap iterations)" },
    });
  }

  $("experience-question").addEventListener("input", updateQuestion);
  updateQuestion();
  ["median", "mean"].forEach((stat) => ["FE", "LW", "SF"].forEach((group) => plotScatter(stat, group)));
  plotNe("extremes", "ne-extremes-table", "ne-extremes-chart", "Fraction not equal to 1 or 99");
  plotNe("rounded", "ne-rounded-table", "ne-rounded-chart", "Fraction not rounded to nearest 5");
  plotNe("roundedAndExtremes", "ne-rounded-extremes-table", "ne-rounded-extremes-chart", "Fraction neither rounded nor extreme");
}

async function renderAggregation() {
  const data = await getJson("./data/aggregation.json");
  $("app").innerHTML =
    pageTitle(
      "Can we aggregate Blind Mode predictions to generate more accurate predictions?",
      `<p class="lead">Aggregations of predictions are often more accurate than individual predictions. Since we have the predictions of many participants in the Blind Mode, we can use this data to generate our own aggregated prediction.</p>
       <p>The straightforward approach is to take a mean or median, but those approaches have a centralizing tendency and do not account for the fact that some participants are better predictors than others.</p>`,
    ) +
    section(
      "Beta transformed arithmetic mean",
      `<p><a href="https://doi.org/10.1371/journal.pone.0256919" target="_blank" rel="noreferrer">Hanea et al., 2021</a> found that the beta transformed arithmetic mean outperformed other aggregation methods in their forecasting data sets. The BetaArMean calculates the mean prediction and then transforms it using the cumulative distribution function of a beta distribution, effectively extremising the aggregate.</p>
       ${chartDiv("aggregation-beta-scatter")}`,
    ) +
    section(
      "Beta transformed experience-weighted arithmetic mean",
      `<p>Although the mean predictions did not systematically differ much across groups, there were differences in individual predictions. I shook my magic 8-ball and used weights of [0.05, 0.8, 0.1, 0.05] for [All, SF, FE, LW] participants, with toned-down extremization.</p>
       ${chartDiv("aggregation-weighted-scatter")}
       <p>Predictions must be integer values between 1 and 99, which leaves the following predictions:</p>
       ${tableShell("aggregation-predictions-table")}`,
    );
  plot("aggregation-beta-scatter", [
    { type: "scatter", mode: "markers", x: data.betaScatter.x, y: data.betaScatter.y, marker: { color: "#0b7261", size: 8 }, name: "Questions" },
    { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: "#8b948f", dash: "dash" }, name: "Equality" },
  ], {
    title: "Mean prediction vs beta transformed value",
    xaxis: { title: "Mean prediction", range: [0, 1] },
    yaxis: { title: "Beta transformed values", range: [0, 1] },
  });
  plot("aggregation-weighted-scatter", [
    { type: "scatter", mode: "markers", x: data.weightedScatter.x, y: data.weightedScatter.y, marker: { color: "#c87929", size: 8 }, name: "Questions" },
    { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: "#8b948f", dash: "dash" }, name: "Equality" },
  ], {
    title: "Superforecaster mean vs weighted transformed prediction",
    xaxis: { title: "Superforecaster mean", range: [0, 1] },
    yaxis: { title: "Weighted transformed prediction", range: [0, 1] },
  });
  renderTable("aggregation-predictions-table", data.myPredictionsTable, {
    pageSize: 50,
    longColumns: ["Question text"],
  });
}

async function renderSimulations() {
  const data = await getJson("./data/simulations.json");
  $("app").innerHTML =
    pageTitle(
      "How do our aggregate predictions fare in Monte Carlo simulations of possible futures?",
      `<p class="lead">While we do not know the future, we can simulate it. We can take the probability of an event occurring and simulate many possible futures, then evaluate the distribution of forecasting scores caused by randomness.</p>
       <p>The scoring method that would be used for the contest was not given, so this analysis uses the Brier score, where lower is better.</p>
       <div class="equation">Brier = (1 / N) * sum((f_i - o_i)^2)</div>`,
    ) +
    section(
      "Aggregated predictions vs the Super forecasters",
      `<h3>What if the mean Super forcaster predictions were correctly calibrated?</h3>
       <p>Here are simulation results from assuming that the mean predictions of the Super forecaster group are the true underlying probabilities of the events.</p>
       <button id="sim-sf-button">Run another precomputed simulation</button>
       ${chartDiv("sim-sf-score")}
       ${chartDiv("sim-sf-percentile")}
       <p id="sim-sf-summary"></p>
       <h3>What if the aggregated predictions had perfect calibration?</h3>
       <p>Here are results from assuming that the aggregated predictions are the true underlying probabilities:</p>
       <button id="sim-perfect-button">Run another precomputed simulation</button>
       ${chartDiv("sim-perfect-score")}
       ${chartDiv("sim-perfect-percentile")}
       <p id="sim-perfect-summary"></p>`,
    ) +
    section(
      "Aggregated predictions vs Blind Mode predictions",
      `<p>How do the aggregated predictions fare against all Blind Mode participants' predictions in Monte Carlo simulations of possible futures?</p>
       <button id="sim-blind-button">Run another precomputed simulation</button>
       ${chartDiv("sim-blind-score")}
       ${chartDiv("sim-blind-percentile")}
       <p id="sim-blind-summary"></p>`,
    );

  function plotScoreDist(id, variant) {
    plot(id, [
      { type: "scatter", mode: "lines", name: "Base Brier Scores", x: variant.scoreDistribution.base.x, y: variant.scoreDistribution.base.y, line: { color: "#8b948f" } },
      { type: "scatter", mode: "lines", name: "My Brier Scores", x: variant.scoreDistribution.mine.x, y: variant.scoreDistribution.mine.y, line: { color: "#0b7261" } },
    ], {
      title: `Brier score distribution (seed ${variant.seed})`,
      xaxis: { title: "Brier score" },
      yaxis: { title: "Probability density" },
    });
  }
  function plotPercentile(id, variant, title) {
    plotHistogram(id, variant.percentileHistogram, title, {
      x: "My Brier score percentile",
      y: "Probability density",
    });
  }
  function renderSf() {
    const variant = data.superforecasterCalibration[pageState.simSf];
    plotScoreDist("sim-sf-score", variant);
    plotPercentile("sim-sf-percentile", variant, "Aggregate prediction percentile");
    $("sim-sf-summary").textContent = `The median Brier score percentile of the aggreagate predictions is ${fmtNumber(variant.medianPercentile, 2)}.`;
  }
  function renderPerfect() {
    const variant = data.perfectCalibration[pageState.simPerfect];
    plotScoreDist("sim-perfect-score", variant);
    plotPercentile("sim-perfect-percentile", variant, "Aggregate prediction percentile");
    $("sim-perfect-summary").textContent = `The median Brier score percentile of the aggreagate predictions is ${fmtNumber(variant.medianPercentile, 2)}.`;
  }
  function renderBlind() {
    const variant = data.blindMode[pageState.simBlind];
    plotHistogram("sim-blind-score", variant.blindModeHistogram, `Blind Mode score distribution (seed ${variant.seed})`, {
      x: "Brier score (lower is better)",
      y: "Count (Blind Mode participants)",
    });
    Plotly.addTraces("sim-blind-score", [
      { type: "scatter", mode: "lines", name: "Aggregate prediction", x: [variant.myBrierScore, variant.myBrierScore], y: [0, Math.max(...variant.blindModeHistogram.y)], line: { color: "#b64242" } },
    ]);
    plotHistogram("sim-blind-percentile", variant.percentileHistogram, "Aggregate Brier score percentile", {
      x: "Brier score percentile (lower is better)",
      y: "Count (simulations)",
    });
    $("sim-blind-summary").innerHTML = `The mean percentile of the aggregate predictions is ${fmtNumber(variant.meanPercentile, 2)}%. That corresponds to finishing in ${variant.finishPlace} place in a field of ${variant.participantCount} participants. In only ${fmtNumber(variant.winPercent, 2)}% of the simulations do the aggregate predictions win, even though they are <strong>perfectly calibrated</strong>.`;
  }
  $("sim-sf-button").addEventListener("click", () => {
    pageState.simSf = (pageState.simSf + 1) % data.superforecasterCalibration.length;
    renderSf();
  });
  $("sim-perfect-button").addEventListener("click", () => {
    pageState.simPerfect = (pageState.simPerfect + 1) % data.perfectCalibration.length;
    renderPerfect();
  });
  $("sim-blind-button").addEventListener("click", () => {
    pageState.simBlind = (pageState.simBlind + 1) % data.blindMode.length;
    renderBlind();
  });
  renderSf();
  renderPerfect();
  renderBlind();
}

function isoTimes(times) {
  return times.map((time) => (time == null ? null : new Date(time).toISOString()));
}

function sampleMarketSeries(series, includeAll) {
  const step = includeAll ? 1 : 10;
  const time = [];
  const probability = [];
  for (let index = 0; index < series.time.length; index += step) {
    time.push(series.time[index]);
    probability.push(series.probability[index]);
  }
  const last = series.time.length - 1;
  if (time.at(-1) !== series.time[last]) {
    time.push(series.time[last]);
    probability.push(series.probability[last]);
  }
  return { time, probability };
}

async function renderMarkets() {
  const data = await getJson("./data/markets.json");
  if (pageState.marketPrediction == null) {
    pageState.marketPrediction = Math.round(data.myPredictions[pageState.marketQuestion - 1] * 100);
  }
  $("app").innerHTML =
    pageTitle(
      "How well do our aggregate predictions compare to prediction markets across the year?",
      `<p class="lead"><a href="https://manifold.markets/" target="_blank" rel="noreferrer">Manifold Markets</a> had a prediction market for each question. This page evaluates the aggregate predictions against market histories across time.</p>
       <p>The analysis uses both Brier score and root mean squared error (RMSE). Lower values indicate more accurate predictions. The contest has now concluded and the Metaculus scoring function was used rather than Brier score.</p>`,
    ) +
    section(
      "Market time series",
      `<p>Select a question and a prediction to see the time series of the market.</p>
       <div class="control-row">
        <div class="control"><label for="market-question">Select question <output id="market-question-output">${pageState.marketQuestion}</output></label><input id="market-question" type="range" min="1" max="50" value="${pageState.marketQuestion}" /></div>
        <div class="control"><label for="market-prediction">Make a prediction <output id="market-prediction-output">${pageState.marketPrediction}</output></label><input id="market-prediction" type="range" min="1" max="99" value="${pageState.marketPrediction}" /></div>
        <label class="control"><span>Plot all time points</span><input id="market-all" type="checkbox" ${pageState.marketAllPoints ? "checked" : ""} /></label>
       </div>
       <p id="market-question-text"></p>
       ${chartDiv("market-probability", "small")}
       ${chartDiv("market-rmse", "small")}
       ${chartDiv("market-brier", "small")}`,
    ) +
    section(
      "Aggregated predictions",
      `<p>How did our aggregated predictions do over time? The scores increase over time, as the market incorporates new information that was not available to the Blind Mode participants at the start of the contest.</p>
       ${chartDiv("market-aggregate-mse", "small")}
       ${chartDiv("market-aggregate-brier", "small")}`,
    ) +
    section(
      "Final scores",
      `<p>How did our aggregated predictions do against Blind Mode participants at the end of the contest?</p>
       ${chartDiv("market-violin")}
       <p>Our aggregate predictions had a final Brier score of ${fmtNumber(data.summary.aggregatedFinalBrier, 3)}. The top 10 Blind Mode participants had final Brier scores of ${escapeHtml(JSON.stringify(data.summary.topTenBrier))}. The aggregate score would have placed ${data.summary.myRank}, which is the top ${fmtPercent(data.summary.myPercentile, 2)}.</p>
       <p>The median score of the Superforecasters was ${fmtNumber(data.summary.superforecasterMedian, 3)}. This would have placed them at ${data.summary.superforecasterRank}, which is the ${fmtPercent(data.summary.superforecasterPercentile, 2)}-tile ranking.</p>`,
    ) +
    section(
      "Final scores of all Blind Mode participants",
      `<p>Note: I used the Brier score to measure prediction accuracy, but the final results of the contest were based on the Metaculus scoring function.</p>
       <button id="load-final-table" class="secondary">Load final participant score table</button>
       <div id="market-final-table"></div>
       <p>To see detailed time series analysis of the blind mode participants see the notebook <a href="https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/3_manifold.ipynb" target="_blank" rel="noreferrer">./notebooks/3_manifold.ipynb</a>.</p>`,
    );

  async function updateMarket() {
    const q = Number($("market-question").value);
    const predPercent = Number($("market-prediction").value);
    pageState.marketQuestion = q;
    pageState.marketPrediction = predPercent;
    pageState.marketAllPoints = $("market-all").checked;
    $("market-question-output").textContent = q;
    $("market-prediction-output").textContent = predPercent;
    const question = questionByNumber(data.questions, q);
    $("market-question-text").innerHTML = `Question: ${escapeHtml(question.text)}<br />Actual outcome: <span class="outcome ${question.outcome.toLowerCase()}">${question.outcome}</span>`;
    const raw = await getJson(`./data/markets/q${q}.json`);
    const sampled = sampleMarketSeries(raw, pageState.marketAllPoints);
    const x = isoTimes(sampled.time);
    const prediction = predPercent / 100;
    const rmse = sampled.probability.map((value) => (value == null ? null : Math.sqrt((prediction - value) ** 2)));
    const brier = sampled.probability.map((value) => (value == null ? null : Math.sqrt((prediction - Math.round(value)) ** 2)));
    plot("market-probability", [{ type: "scatter", mode: "lines", x, y: sampled.probability, line: { color: "#0b7261" }, name: "Market probability" }], {
      title: "Market probability",
      xaxis: { title: "Time" },
      yaxis: { title: "Probability", range: [0, 1] },
    });
    plot("market-rmse", [{ type: "scatter", mode: "lines", x, y: rmse, line: { color: "#c87929" }, name: "RMSE" }], {
      title: "RMSE",
      xaxis: { title: "Time" },
      yaxis: { title: "RMSE", range: [0, 1] },
    });
    plot("market-brier", [{ type: "scatter", mode: "lines", x, y: brier, line: { color: "#b64242" }, name: "Brier score" }], {
      title: "Brier score",
      xaxis: { title: "Time" },
      yaxis: { title: "Brier Score", range: [0, 1] },
    });
  }

  function renderScoreSeries() {
    const x = isoTimes(data.scoreTimeSeries.time);
    plot("market-aggregate-mse", [{ type: "scatter", mode: "lines", x, y: data.scoreTimeSeries.mse, line: { color: "#0b7261" }, name: "MSE" }], {
      title: "Aggregate prediction MSE vs markets",
      xaxis: { title: "Time" },
      yaxis: { title: "MSE" },
    });
    plot("market-aggregate-brier", [{ type: "scatter", mode: "lines", x, y: data.scoreTimeSeries.brier, line: { color: "#c87929" }, name: "Brier" }], {
      title: "Aggregate prediction Brier score vs markets",
      xaxis: { title: "Time" },
      yaxis: { title: "Brier Score" },
    });
  }

  function renderViolin() {
    const traces = [
      ["all", "All"],
      ["FE", "Forecasting experience"],
      ["LW", "LessWrong"],
      ["SF", "Superforecaster"],
    ].map(([key, label]) => ({
      type: "violin",
      y: data.violin[key],
      name: label,
      box: { visible: true },
      meanline: { visible: true },
      points: "all",
      marker: { size: 2, color: groupColors[key] },
      line: { color: groupColors[key] },
    }));
    plot("market-violin", traces, { title: "Final Brier scores by group", yaxis: { title: "Brier score" } });
  }

  $("market-question").addEventListener("input", () => {
    const q = Number($("market-question").value);
    const defaultPrediction = Math.round(data.myPredictions[q - 1] * 100);
    $("market-prediction").value = defaultPrediction;
    pageState.marketPrediction = defaultPrediction;
    updateMarket();
  });
  $("market-prediction").addEventListener("input", updateMarket);
  $("market-all").addEventListener("change", updateMarket);
  $("load-final-table").addEventListener("click", async () => {
    $("market-final-table").innerHTML = '<p class="loading">Loading final table...</p>';
    const table = await getJson("./data/market-final-table.json");
    renderTable("market-final-table", table, { pageSize: 20 });
    $("load-final-table").remove();
  });
  renderScoreSeries();
  renderViolin();
  updateMarket();
}

async function renderPosthoc() {
  const data = await getJson("./data/posthoc.json");
  $("app").innerHTML =
    pageTitle(
      "How close to optimal were our aggregation parameters?",
      `<p class="lead">Now that the contest is over, we can evaluate how our aggregated predictions would have fared with alternate aggregation parameters. This page uses Brier score, where lower is better.</p>`,
    ) +
    section(
      "Fixed weights and varying beta parameters",
      `<p>This plot shows the Brier score of the aggregated predictions as a function of beta parameters, with group weights held constant at their original values.</p>
       ${chartDiv("posthoc-beta-1d", "small")}
       <p>Same as above, but now allowing the beta parameters to vary independently.</p>
       ${chartDiv("posthoc-beta-2d", "tall")}`,
    ) +
    section(
      "Fixed beta parameters and varying weights",
      `<p>For these figures beta_a = beta_b = 1/3 and group weights vary across a 4D grid, considering only parameter sets where the weights sum to 1.</p>
       ${chartDiv("posthoc-score-hist", "small")}
       <p>Dimensionality reduction of the 4d grid of weights using t-SNE. The Brier score of each parameter set is represented by point color.</p>
       ${chartDiv("posthoc-tsne-2d", "tall")}
       <p>1D t-SNE of the parameter mesh when excluding one of the group weights (SF, left, and FE, right).</p>
       <div class="two-col">${chartDiv("posthoc-tsne-sf", "small")}${chartDiv("posthoc-tsne-fe", "small")}</div>
       <p>Additional notebook results, including PCA and global parameter optimization, are in <a href="https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/4_post_hoc_aggregation.ipynb" target="_blank" rel="noreferrer">./notebooks/4_post_hoc_aggregation.ipynb</a>. By optimizing over the entire parameter space, a minimal Brier Score of 0.147730 was found with fitted parameters of 0.000000, 0.722385, 0.082014, 0.195602, 0.227142, 0.276849.</p>`,
    );
  plot("posthoc-beta-1d", [{ type: "scatter", mode: "lines", x: data.beta1d.x, y: data.beta1d.y, line: { color: "#0b7261" } }], {
    title: "Brier score vs equal beta parameters",
    xaxis: { title: "beta_a = beta_b", type: "log" },
    yaxis: { title: "Brier score" },
  });
  plot("posthoc-beta-2d", [{ type: "heatmap", x: data.beta2d.x, y: data.beta2d.y, z: data.beta2d.z, colorscale: "Viridis", colorbar: { title: "Brier" } }], {
    title: "Brier score vs beta_a and beta_b",
    xaxis: { title: "beta_b", type: "log" },
    yaxis: { title: "beta_a", type: "log" },
  });
  plotHistogram("posthoc-score-hist", data.weightScoreHistogram, "All Brier scores across the 4D grid of weights", {
    x: "Brier score",
    y: "Count",
  });
  plot("posthoc-tsne-2d", [{ type: "scatter", mode: "markers", x: data.tsne2d.x, y: data.tsne2d.y, marker: { size: 5, color: data.tsne2d.score, colorscale: "Viridis", colorbar: { title: "Brier" } } }], {
    title: "2D t-SNE of parameter mesh",
    xaxis: { title: "TSNE 1" },
    yaxis: { title: "TSNE 2" },
  });
  plot("posthoc-tsne-sf", [{ type: "scatter", mode: "markers", x: data.tsne1d.sfWeight, y: data.tsne1d.minusSf, marker: { size: 5, color: data.tsne1d.score, colorscale: "Viridis" } }], {
    title: "Excluding SF weight",
    xaxis: { title: "SF weights" },
    yaxis: { title: "TSNE 1" },
  });
  plot("posthoc-tsne-fe", [{ type: "scatter", mode: "markers", x: data.tsne1d.feWeight, y: data.tsne1d.minusFe, marker: { size: 5, color: data.tsne1d.score, colorscale: "Viridis" } }], {
    title: "Excluding FE weight",
    xaxis: { title: "FE weights" },
    yaxis: { title: "TSNE 1" },
  });
}

async function renderSupervised() {
  const data = await getJson("./data/supervised.json");
  if (!pageState.featureSlug) pageState.featureSlug = data.defaultFeatureSlug;
  $("app").innerHTML =
    pageTitle(
      "What attributes of Blind Mode participants are most predictive of accurate forecasting?",
      `<p class="lead">Now that the contest is over, we can evaluate which Blind Mode participants performed best. Then we can ask if there are identifiable characteristics of the best-performing participants that could generate an improved aggregate forecast.</p>
       <p>See the Simulating outcomes page for a discussion of Brier score. Lower is better.</p>`,
    ) +
    section(
      "Feature visualization",
      `<div class="control-row">
        <div class="control"><label for="feature-select">Select a feature to plot</label><select id="feature-select"></select></div>
        <button id="feature-random">Randomize feature</button>
       </div>
       ${chartDiv("feature-chart")}`,
    ) +
    section(
      "Feature correlations",
      `<p>We can also look at the correlation between each feature and the score. A volcano plot shows correlation coefficient on the x-axis and p-value on the y-axis. Since a lower Brier score is better, positive correlations are associated with worse scores and negative correlations are associated with better scores.</p>
       ${chartDiv("feature-volcano")}
       ${tableShell("feature-correlation-table")}`,
    ) +
    section(
      "XGBoost model",
      `<p>See <a href="https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_xgb.ipynb" target="_blank" rel="noreferrer">./notebooks/5_post_hoc_xgb.ipynb</a> for ongoing analysis and results that will be transfered here, where an XGBoost model is used to predict a participants' Brier score based on their survey question answers.</p>`,
    ) +
    section(
      "Aggregation from model",
      `<p>The trained model can then be used to perform prediction aggregation on participants that were not part of the training set.</p>`,
    ) +
    section(
      "Neural network model",
      `<p>See <a href="https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_nn.ipynb" target="_blank" rel="noreferrer">./notebooks/5_post_hoc_nn.ipynb</a> for results similar to the XGBoost model, but using a neural network.</p>
       <p>See <a href="https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/5_post_hoc_features.ipynb" target="_blank" rel="noreferrer">./notebooks/5_post_hoc_features.ipynb</a> for addidional analysis and results.</p>
       <p>See <a href="https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/6_hyperopt_xgb.ipynb" target="_blank" rel="noreferrer">./notebooks/6_hyperopt_xgb.ipynb</a> and <a href="https://github.com/jbreffle/acx-prediction-contest/blob/main/notebooks/6_hyperopt_nn.ipynb" target="_blank" rel="noreferrer">./notebooks/6_hyperopt_nn.ipynb</a> for hyperparameter optimization of the models.</p>`,
    );

  const select = $("feature-select");
  select.innerHTML = data.features.map((feature) => `<option value="${feature.slug}">${escapeHtml(feature.name)}</option>`).join("");
  select.value = pageState.featureSlug;

  async function renderFeature(slug) {
    pageState.featureSlug = slug;
    select.value = slug;
    const feature = await getJson(`./data/features/${slug}.json`);
    if (feature.type === "continuous") {
      plot("feature-chart", [{ type: "scatter", mode: "markers", x: feature.x, y: feature.y, marker: { size: 5, color: "#0b7261", opacity: 0.62 } }], {
        title: feature.feature,
        xaxis: { title: feature.feature },
        yaxis: { title: "Brier score" },
      });
    } else {
      const traces = feature.groups.map((group) => ({
        type: "violin",
        y: group.scores,
        name: group.label,
        box: { visible: true },
        meanline: { visible: true },
        points: "all",
        marker: { size: 3 },
      }));
      plot("feature-chart", traces, { title: feature.feature, yaxis: { title: "Brier score" } });
    }
  }

  function renderVolcano() {
    const traces = ["better", "worse", "not significant"].map((effect) => {
      const points = data.volcano.filter((point) => point.effect === effect);
      return {
        type: "scatter",
        mode: "markers",
        name: effect,
        x: points.map((point) => point.r_value),
        y: points.map((point) => point.neg_log10_p),
        text: points.map((point) => point.feature),
        marker: { color: groupColors[effect], size: 8 },
      };
    });
    plot("feature-volcano", traces, {
      title: "Feature correlations to Brier score",
      xaxis: { title: "r-value" },
      yaxis: { title: "-log10 p-value" },
    });
  }

  select.addEventListener("change", () => renderFeature(select.value));
  $("feature-random").addEventListener("click", () => {
    const feature = data.features[Math.floor(Math.random() * data.features.length)];
    renderFeature(feature.slug);
  });
  renderFeature(pageState.featureSlug);
  renderVolcano();
  renderTable("feature-correlation-table", {
    columns: ["feature", "r_value", "p_value", "effect"],
    rows: data.correlations.map((row) => [row.feature, row.r_value, row.p_value, row.effect]),
  }, { pageSize: 20 });
}

async function route() {
  const key = (window.location.hash || "#home").slice(1);
  const active = routes.some(([routeKey]) => routeKey === key) ? key : "home";
  renderNav(active);
  $("app").innerHTML = '<p class="loading">Loading...</p>';
  try {
    if (active === "home") await renderHome();
    if (active === "experience") await renderExperience();
    if (active === "aggregation") await renderAggregation();
    if (active === "simulations") await renderSimulations();
    if (active === "markets") await renderMarkets();
    if (active === "posthoc") await renderPosthoc();
    if (active === "supervised") await renderSupervised();
    window.scrollTo(0, 0);
  } catch (error) {
    console.error(error);
    $("app").innerHTML = `<section class="section"><h1>Load error</h1><p>${escapeHtml(error.message)}</p></section>`;
  }
}

window.addEventListener("hashchange", route);
route();
