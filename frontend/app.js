const apiBaseUrl = String(window.DECISION_ASSISTANT_CONFIG?.apiBaseUrl || "").replace(/\/$/, "");

const state = {
  authMode: "login",
  assistantMode: "assistant",
  trainMode: "forecast",
  resultsMode: "forecast",
  resultsTab: "summary",
  token: localStorage.getItem("decision-intel-token") || "",
  user: JSON.parse(localStorage.getItem("decision-intel-user") || "null"),
  activeView: localStorage.getItem("decision-intel-view") || "auth",
  datasets: [],
  documents: [],
  runs: [],
  classificationRuns: [],
  recentTasks: [],
  assistantPending: false,
  uploadHeaders: [],
  dashboardSummary: null,
  lastSavedMessage: "Nothing saved yet",
  lastAssistantSources: [],
};

const elements = {
  pageHeading: document.getElementById("pageHeading"),
  pageNav: document.getElementById("pageNav"),
  navButtons: [...document.querySelectorAll(".nav-btn")],
  pageViews: [...document.querySelectorAll(".page-view")],
  navFootnote: document.getElementById("navFootnote"),
  sessionStatus: document.getElementById("sessionStatus"),
  lastSavedStatus: document.getElementById("lastSavedStatus"),
  logoutBtn: document.getElementById("logoutBtn"),
  authForm: document.getElementById("authForm"),
  authMode: document.getElementById("authMode"),
  fullNameGroup: document.getElementById("fullNameGroup"),
  authSignedInState: document.getElementById("authSignedInState"),
  authSignedInCopy: document.getElementById("authSignedInCopy"),
  authContinueBtn: document.getElementById("authContinueBtn"),
  authSubmit: document.getElementById("authSubmit"),
  authStatus: document.getElementById("authStatus"),
  refreshDashboard: document.getElementById("refreshDashboard"),
  metricsGrid: document.getElementById("metricsGrid"),
  overviewSummary: document.getElementById("overviewSummary"),
  hostingSummary: document.getElementById("hostingSummary"),
  recentTaskList: document.getElementById("recentTaskList"),
  csvUploadForm: document.getElementById("csvUploadForm"),
  csvFileInput: document.getElementById("csvFileInput"),
  uploadTargetSelect: document.getElementById("uploadTargetSelect"),
  uploadTimeSelect: document.getElementById("uploadTimeSelect"),
  uploadCsvHelper: document.getElementById("uploadCsvHelper"),
  csvStatus: document.getElementById("csvStatus"),
  datasetTableHead: document.querySelector("#datasetTable thead"),
  datasetTableBody: document.querySelector("#datasetTable tbody"),
  workflowGuide: document.getElementById("workflowGuide"),
  datasetList: document.getElementById("datasetList"),
  docUploadForm: document.getElementById("docUploadForm"),
  docStatus: document.getElementById("docStatus"),
  documentList: document.getElementById("documentList"),
  trainMode: document.getElementById("trainMode"),
  trainRecommendation: document.getElementById("trainRecommendation"),
  forecastTrainPanel: document.getElementById("forecastTrainPanel"),
  classificationTrainPanel: document.getElementById("classificationTrainPanel"),
  trainRunSummary: document.getElementById("trainRunSummary"),
  datasetSelect: document.getElementById("datasetSelect"),
  forecastTargetInput: document.getElementById("forecastTargetInput"),
  forecastTimeInput: document.getElementById("forecastTimeInput"),
  forecastForm: document.getElementById("forecastForm"),
  forecastStatus: document.getElementById("forecastStatus"),
  classificationDatasetSelect: document.getElementById("classificationDatasetSelect"),
  classificationTargetInput: document.getElementById("classificationTargetInput"),
  classificationForm: document.getElementById("classificationForm"),
  classificationStatus: document.getElementById("classificationStatus"),
  resultsMode: document.getElementById("resultsMode"),
  resultsTab: document.getElementById("resultsTab"),
  forecastSummaryPanel: document.getElementById("forecastSummaryPanel"),
  forecastChartsPanel: document.getElementById("forecastChartsPanel"),
  forecastWhyPanel: document.getElementById("forecastWhyPanel"),
  forecastTryPanel: document.getElementById("forecastTryPanel"),
  classificationSummaryPanel: document.getElementById("classificationSummaryPanel"),
  classificationChartsPanel: document.getElementById("classificationChartsPanel"),
  classificationWhyPanel: document.getElementById("classificationWhyPanel"),
  classificationTryPanel: document.getElementById("classificationTryPanel"),
  forecastMetrics: document.getElementById("forecastMetrics"),
  runSelect: document.getElementById("runSelect"),
  runLibrary: document.getElementById("runLibrary"),
  refreshRun: document.getElementById("refreshRun"),
  predictAgain: document.getElementById("predictAgain"),
  forecastChart: document.getElementById("forecastChart"),
  globalShapChart: document.getElementById("globalShapChart"),
  localShapChart: document.getElementById("localShapChart"),
  shapInsights: document.getElementById("shapInsights"),
  forecastProjection: document.getElementById("forecastProjection"),
  classificationMetrics: document.getElementById("classificationMetrics"),
  classificationRunSelect: document.getElementById("classificationRunSelect"),
  classificationRunLibrary: document.getElementById("classificationRunLibrary"),
  refreshClassificationRun: document.getElementById("refreshClassificationRun"),
  classificationImportanceChart: document.getElementById("classificationImportanceChart"),
  classificationConfusionChart: document.getElementById("classificationConfusionChart"),
  classificationInsights: document.getElementById("classificationInsights"),
  classificationPredictForm: document.getElementById("classificationPredictForm"),
  classificationFeatureFields: document.getElementById("classificationFeatureFields"),
  classificationPrediction: document.getElementById("classificationPrediction"),
  assistantMode: document.getElementById("assistantMode"),
  assistantContextChips: document.getElementById("assistantContextChips"),
  assistantForm: document.getElementById("assistantForm"),
  assistantSubmit: document.getElementById("assistantSubmit"),
  assistantSourceList: document.getElementById("assistantSourceList"),
  conversation: document.getElementById("conversation"),
};

const protectedViews = new Set(["overview", "data", "forecast", "classification", "assistant"]);
const viewTitles = {
  auth: "Your Account",
  overview: "Start",
  data: "Upload",
  forecast: "Train",
  classification: "Results",
  assistant: "Ask AI",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNumber(value, digits = 3) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "-";
}

function setImageSource(node, source) {
  if (!node) {
    return;
  }
  if (source) {
    node.src = source;
  } else {
    node.removeAttribute("src");
  }
}

function setStatus(node, message = "", isError = false) {
  if (!node) {
    return;
  }
  node.textContent = message;
  node.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function rememberLastSaved(message) {
  state.lastSavedMessage = message;
  elements.lastSavedStatus.textContent = message;
  elements.lastSavedStatus.style.color = "var(--text)";
}

function getDatasetById(datasetId) {
  return state.datasets.find((item) => item.id === Number(datasetId)) || null;
}

function getForecastRunById(runId) {
  return state.runs.find((item) => item.id === Number(runId)) || null;
}

function getClassificationRunById(runId) {
  return state.classificationRuns.find((item) => item.id === Number(runId)) || null;
}

function getDatasetName(datasetId) {
  return getDatasetById(datasetId)?.name || `File ${datasetId}`;
}

function getActiveAssistantDatasetId() {
  return state.resultsMode === "classification"
    ? Number(elements.classificationDatasetSelect.value || 0) || null
    : Number(elements.datasetSelect.value || 0) || null;
}

function getActiveAssistantForecastRunId() {
  return state.resultsMode === "forecast" && elements.runSelect.value
    ? Number(elements.runSelect.value)
    : null;
}

function getActiveAssistantClassificationRunId() {
  return state.resultsMode === "classification" && elements.classificationRunSelect.value
    ? Number(elements.classificationRunSelect.value)
    : null;
}

function getWorkflowLabel(workflow) {
  const labels = {
    forecasting: "Future values",
    regression: "Number prediction",
    classification: "Best match",
    exploration: "Review only",
  };
  return labels[workflow] || "Review only";
}

function getTargetKindLabel(kind) {
  const labels = {
    numeric: "Number",
    categorical: "Label",
    missing: "Not chosen",
  };
  return labels[kind] || "Unknown";
}

function describeDataset(dataset) {
  const workflow = dataset.metadata_json?.recommended_workflow || "exploration";
  return `${dataset.name} (${dataset.row_count} rows | ${getWorkflowLabel(workflow)})`;
}

function inferLikelyTarget(columns) {
  const candidates = ["target", "label", "yield", "sales", "demand", "class"];
  const match = columns.find((column) => candidates.includes(column.toLowerCase()));
  return match || "";
}

function inferLikelyTime(columns) {
  const candidates = ["date", "timestamp", "time", "ds"];
  const match = columns.find((column) => candidates.includes(column.toLowerCase()));
  return match || "";
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let insideQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const next = line[index + 1];

    if (character === '"' && insideQuotes && next === '"') {
      current += '"';
      index += 1;
      continue;
    }

    if (character === '"') {
      insideQuotes = !insideQuotes;
      continue;
    }

    if (character === "," && !insideQuotes) {
      values.push(current.trim());
      current = "";
      continue;
    }

    current += character;
  }

  values.push(current.trim());
  return values.filter((value) => value.length);
}

async function loadCsvHeaders(file) {
  const text = await file.text();
  const firstLine = text.split(/\r?\n/).find((line) => line.trim().length);
  return firstLine ? parseCsvLine(firstLine) : [];
}

function populateSelect(node, values, { blankLabel, preferredValue } = {}) {
  if (!node) {
    return;
  }

  const options = [];
  if (blankLabel !== undefined) {
    options.push(`<option value="">${escapeHtml(blankLabel)}</option>`);
  }
  options.push(
    ...values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
  );
  node.innerHTML = options.join("");

  if (preferredValue && values.includes(preferredValue)) {
    node.value = preferredValue;
  }
}

function setSession() {
  if (state.user) {
    elements.sessionStatus.textContent = state.user.email;
    elements.sessionStatus.style.color = "var(--text)";
    elements.logoutBtn.hidden = false;
    elements.navFootnote.textContent = "You are signed in. Move through Start, Upload, Train, Results, and Ask AI.";
    elements.authMode.hidden = true;
    elements.authForm.hidden = true;
    elements.authSignedInState.hidden = false;
    elements.authSignedInCopy.textContent = `You are signed in with ${state.user.email}. Go to Start to pick up where you left off.`;
  } else {
    elements.sessionStatus.textContent = "Not signed in";
    elements.sessionStatus.style.color = "var(--muted)";
    elements.logoutBtn.hidden = true;
    elements.navFootnote.textContent = "Sign in to use the rest of the workspace.";
    elements.authMode.hidden = false;
    elements.authForm.hidden = false;
    elements.authSignedInState.hidden = true;
    elements.authSignedInCopy.textContent = "";
  }
  syncNavigationState();
}

function syncNavigationState() {
  elements.navButtons.forEach((button) => {
    const isProtected = protectedViews.has(button.dataset.view);
    button.disabled = isProtected && !state.token;
  });
}

function setActiveView(view) {
  const resolvedView = !state.token && protectedViews.has(view) ? "auth" : view;
  state.activeView = resolvedView;
  localStorage.setItem("decision-intel-view", resolvedView);

  elements.pageHeading.textContent = viewTitles[resolvedView];
  elements.navButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.view === resolvedView);
  });
  elements.pageViews.forEach((section) => {
    section.hidden = section.dataset.view !== resolvedView;
  });
  window.scrollTo({ top: 0, behavior: "auto" });
}

function toggleAuthMode(mode) {
  state.authMode = mode;
  [...elements.authMode.querySelectorAll("button")].forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  elements.fullNameGroup.hidden = mode !== "register";
  elements.authSubmit.textContent = mode === "register" ? "Create account" : "Sign in";
}

function toggleAssistantMode(mode) {
  state.assistantMode = mode;
  [...elements.assistantMode.querySelectorAll("button")].forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  renderAssistantContext();
}

function toggleTrainMode(mode) {
  state.trainMode = mode;
  [...elements.trainMode.querySelectorAll("button")].forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  const forecastActive = mode === "forecast";
  elements.forecastTrainPanel.hidden = !forecastActive;
  elements.classificationTrainPanel.hidden = forecastActive;
}

function toggleResultsMode(mode) {
  state.resultsMode = mode;
  [...elements.resultsMode.querySelectorAll("button")].forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  syncResultPanels();
  renderAssistantContext();
}

function toggleResultsTab(tab) {
  state.resultsTab = tab;
  [...elements.resultsTab.querySelectorAll("button")].forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  syncResultPanels();
}

function syncResultPanels() {
  const panelMap = {
    forecast: {
      summary: elements.forecastSummaryPanel,
      charts: elements.forecastChartsPanel,
      why: elements.forecastWhyPanel,
      try: elements.forecastTryPanel,
    },
    classification: {
      summary: elements.classificationSummaryPanel,
      charts: elements.classificationChartsPanel,
      why: elements.classificationWhyPanel,
      try: elements.classificationTryPanel,
    },
  };

  Object.values(panelMap).forEach((group) => {
    Object.values(group).forEach((panel) => {
      panel.hidden = true;
    });
  });

  panelMap[state.resultsMode][state.resultsTab].hidden = false;
}

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) {
    headers.set("Authorization", `Bearer ${state.token}`);
  }

  const response = await fetch(`${apiBaseUrl}${path}`, { ...options, headers });
  if (!response.ok) {
    let detail = "Something went wrong";
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch (error) {
      detail = response.statusText;
    }
    throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
  }
  return response.json();
}

function renderMetrics(
  totals = { datasets: 0, documents: 0, forecast_runs: 0, classification_runs: 0, users: 0 }
) {
  const cards = [
    ["Data files", totals.datasets || 0],
    ["Reports", totals.documents || 0],
    ["Future estimates", totals.forecast_runs || 0],
    ["Recommendations", totals.classification_runs || 0],
    ["Accounts", totals.users || 0],
  ];
  elements.metricsGrid.innerHTML = cards
    .map(
      ([label, value]) => `
        <article class="metric-card">
          <span>${label}</span>
          <strong>${value}</strong>
        </article>
      `
    )
    .join("");
}

function renderOverviewSummary(summary) {
  const parts = [];

  const latestForecast = summary?.latest_forecast;
  if (latestForecast?.evaluation) {
    parts.push(`
      <article class="list-item">
        <strong>Latest future estimate</strong>
        <div class="list-meta">Average miss: ${formatNumber(latestForecast.evaluation.mae)} | Fit score: ${formatNumber(latestForecast.evaluation.r2)}</div>
      </article>
    `);
  }

  const latestClassification = summary?.latest_classification;
  if (latestClassification?.evaluation) {
    parts.push(`
      <article class="list-item">
        <strong>Latest recommendation model</strong>
        <div class="list-meta">Right answers: ${formatNumber(latestClassification.evaluation.accuracy)} | Overall balance: ${formatNumber(latestClassification.evaluation.f1_macro)}</div>
      </article>
    `);
  }

  if (!parts.length) {
    parts.push(`
      <article class="list-item">
        <strong>Upload your first file</strong>
        <div class="list-meta">Start in Upload, then move to Train once the file is ready.</div>
      </article>
    `);
  }

  parts.push(`
    <article class="list-item">
      <strong>Use reports when you have them</strong>
      <div class="list-meta">PDF and text reports help Ask AI answer with real supporting notes.</div>
    </article>
  `);

  elements.overviewSummary.innerHTML = parts.join("");
}

function renderHostingSummary(hostingProfile) {
  const target = hostingProfile?.deploy_target || "huggingface-spaces";
  const freeMode = hostingProfile?.free_mode ? "on" : "off";
  elements.hostingSummary.innerHTML = `
    <article class="list-item">
      <strong>Free mode is ${freeMode}</strong>
      <div class="list-meta">The app is tuned to stay lighter on startup and on long-running tasks.</div>
    </article>
    <article class="list-item">
      <strong>Current deploy target</strong>
      <div class="list-meta">${escapeHtml(target)}</div>
    </article>
    <article class="list-item">
      <strong>What to expect on free hosting</strong>
      <div class="list-meta">The first request after idle time may take longer, and local temporary files can disappear after a restart.</div>
    </article>
  `;
}

function renderRecentTasks() {
  elements.recentTaskList.innerHTML = state.recentTasks.length
    ? state.recentTasks
        .map(
          (task) => `
            <article class="list-item">
              <strong>${escapeHtml(task.title)}</strong>
              <div class="list-meta">${escapeHtml(task.detail || "No extra details")}</div>
              <div class="list-meta">Status: ${escapeHtml(task.status)}${task.target_view ? ` | Best page: ${escapeHtml(viewTitles[task.target_view] || task.target_view)}` : ""}</div>
            </article>
          `
        )
        .join("")
    : '<article class="list-item">No recent activity yet.</article>';
}

function renderDatasetPreview(rows) {
  if (!rows || !rows.length) {
    elements.datasetTableHead.innerHTML = "";
    elements.datasetTableBody.innerHTML = "<tr><td>No file preview yet.</td></tr>";
    return;
  }

  const columns = Object.keys(rows[0]);
  elements.datasetTableHead.innerHTML = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
  elements.datasetTableBody.innerHTML = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((column) => `<td>${escapeHtml(row[column] ?? "")}</td>`)
          .join("")}</tr>`
    )
    .join("");
}

function renderWorkflowGuide() {
  const dataset = state.datasets[0];
  if (!dataset) {
    elements.workflowGuide.innerHTML = `
      <article class="list-item">
        <strong>No file uploaded yet</strong>
        <div class="list-meta">Upload a CSV to see a clear next-step recommendation here.</div>
      </article>
    `;
    return;
  }

  const metadata = dataset.metadata_json || {};
  const summary = metadata.workflow_summary || {};
  const warnings = metadata.warnings || [];
  elements.workflowGuide.innerHTML = `
    <article class="list-item">
      <strong>${escapeHtml(summary.headline || "Your file is ready.")}</strong>
      <div class="list-meta">${escapeHtml(summary.next_step || "Go to Train to continue.")}</div>
    </article>
    <article class="list-item">
      <strong>Detected setup</strong>
      <div class="list-meta">Best use: ${escapeHtml(getWorkflowLabel(metadata.recommended_workflow || "exploration"))} | Main result: ${escapeHtml(dataset.target_column || "-")} | Date column: ${escapeHtml(dataset.time_column || "-")}</div>
    </article>
    ${
      warnings.length
        ? `<article class="list-item"><strong>Things to know</strong><div class="list-meta">${warnings.map((warning) => escapeHtml(warning)).join(" ")}</div></article>`
        : ""
    }
  `;
}

function renderDatasets() {
  const currentForecastId = Number(elements.datasetSelect.value);
  const currentClassificationId = Number(elements.classificationDatasetSelect.value);

  if (!state.datasets.length) {
    elements.datasetList.innerHTML = '<article class="list-item">No files uploaded yet.</article>';
    populateSelect(elements.datasetSelect, [], { blankLabel: "Upload a file first" });
    populateSelect(elements.classificationDatasetSelect, [], { blankLabel: "Upload a file first" });
    populateSelect(elements.forecastTargetInput, [], { blankLabel: "Choose a number column" });
    populateSelect(elements.forecastTimeInput, [], { blankLabel: "No date column" });
    populateSelect(elements.classificationTargetInput, [], { blankLabel: "Choose an answer column" });
    renderWorkflowGuide();
    renderTrainRunSummary();
    renderAssistantContext();
    return;
  }

  const options = state.datasets.map((dataset) => ({ label: describeDataset(dataset), value: String(dataset.id) }));
  elements.datasetSelect.innerHTML = options.map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`).join("");
  elements.classificationDatasetSelect.innerHTML = elements.datasetSelect.innerHTML;

  const preferredForecast =
    getDatasetById(currentForecastId) ||
    state.datasets.find((dataset) => dataset.metadata_json?.forecast_ready) ||
    state.datasets.find((dataset) => dataset.metadata_json?.target_kind === "numeric") ||
    state.datasets[0];
  const preferredClassification =
    getDatasetById(currentClassificationId) ||
    state.datasets.find((dataset) => dataset.metadata_json?.recommended_workflow === "classification") ||
    state.datasets[0];

  elements.datasetSelect.value = String(preferredForecast.id);
  elements.classificationDatasetSelect.value = String(preferredClassification.id);

  elements.datasetList.innerHTML = state.datasets
    .map((dataset) => {
      const metadata = dataset.metadata_json || {};
      return `
        <article class="list-item">
          <strong>${escapeHtml(dataset.name)}</strong>
          <div class="list-meta">${escapeHtml(dataset.filename)} | Rows: ${dataset.row_count}</div>
          <div class="list-meta">Best use: ${escapeHtml(getWorkflowLabel(metadata.recommended_workflow || "exploration"))} | Result type: ${escapeHtml(getTargetKindLabel(metadata.target_kind || "unknown"))}</div>
          <div class="list-meta">Columns: ${escapeHtml((metadata.columns || []).join(", "))}</div>
        </article>
      `;
    })
    .join("");

  renderWorkflowGuide();
  syncForecastDefaults();
  syncClassificationDefaults();
  renderTrainRunSummary();
  renderAssistantContext();
}

function syncForecastDefaults() {
  const dataset = getDatasetById(elements.datasetSelect.value);
  if (!dataset) {
    setStatus(elements.forecastStatus, "Upload a file before creating a future estimate.");
    return;
  }

  const metadata = dataset.metadata_json || {};
  const numericColumns = metadata.numeric_columns || [];
  const allColumns = metadata.columns || [];

  populateSelect(elements.forecastTargetInput, numericColumns, {
    blankLabel: "Choose a number column",
    preferredValue: dataset.target_column && numericColumns.includes(dataset.target_column) ? dataset.target_column : numericColumns[0],
  });
  populateSelect(elements.forecastTimeInput, allColumns, {
    blankLabel: "No date column",
    preferredValue: dataset.time_column || metadata.suggested_time_column,
  });

  if (metadata.recommended_workflow === "classification") {
    setStatus(
      elements.forecastStatus,
      `This file looks better for best-match answers. If you still train here, pick a number such as temperature, humidity, or rainfall.`
    );
    return;
  }

  if (!metadata.forecast_ready && metadata.target_kind === "numeric") {
    setStatus(elements.forecastStatus, "This file has numbers but no clear date column. Row order will be used.");
    return;
  }

  setStatus(elements.forecastStatus, "Ready to create a future estimate.");
}

function syncClassificationDefaults() {
  const dataset = getDatasetById(elements.classificationDatasetSelect.value);
  if (!dataset) {
    setStatus(elements.classificationStatus, "Upload a file before creating a recommendation model.");
    return;
  }

  const metadata = dataset.metadata_json || {};
  const allColumns = metadata.columns || [];
  populateSelect(elements.classificationTargetInput, allColumns, {
    blankLabel: "Choose an answer column",
    preferredValue: dataset.target_column || metadata.suggested_target_column,
  });

  if (metadata.recommended_workflow === "classification") {
    setStatus(
      elements.classificationStatus,
      `This file is ready for best-match predictions. Suggested answer column: "${dataset.target_column || metadata.suggested_target_column || "label"}".`
    );
    return;
  }

  if (metadata.target_kind === "numeric") {
    setStatus(elements.classificationStatus, "This page works best for label-style answers such as class or crop name.");
    return;
  }

  setStatus(elements.classificationStatus, "Choose the answer column you want the app to learn.");
}

function renderTrainRunSummary() {
  const latestForecast = state.runs[0];
  const latestClassification = state.classificationRuns[0];
  const parts = [];

  if (latestForecast) {
    parts.push(`
      <article class="list-item">
        <strong>Latest future estimate</strong>
        <div class="list-meta">${escapeHtml(getDatasetName(latestForecast.dataset_id))} | ${escapeHtml(latestForecast.target_column)} | ${latestForecast.horizon} future steps</div>
      </article>
    `);
  }
  if (latestClassification) {
    parts.push(`
      <article class="list-item">
        <strong>Latest recommendation model</strong>
        <div class="list-meta">${escapeHtml(getDatasetName(latestClassification.dataset_id))} | ${escapeHtml(latestClassification.target_column)}</div>
      </article>
    `);
  }

  if (!parts.length) {
    parts.push('<article class="list-item">No saved models yet.</article>');
  }

  elements.trainRunSummary.innerHTML = parts.join("");
}

function renderRuns() {
  const currentRunId = Number(elements.runSelect.value);
  elements.runSelect.innerHTML = state.runs.length
    ? state.runs
        .map(
          (run) =>
            `<option value="${run.id}">Saved estimate ${run.id} - ${escapeHtml(run.target_column)}</option>`
        )
        .join("")
    : '<option value="">Create an estimate first</option>';

  if (state.runs.length) {
    const selectedRun = state.runs.find((run) => run.id === currentRunId) || state.runs[0];
    elements.runSelect.value = String(selectedRun.id);
  }

  elements.runLibrary.innerHTML = state.runs.length
    ? state.runs
        .map(
          (run) => `
            <article class="list-item">
              <strong>Saved estimate ${run.id}</strong>
              <div class="list-meta">From: ${escapeHtml(getDatasetName(run.dataset_id))}</div>
              <div class="list-meta">Number: ${escapeHtml(run.target_column)} | Steps ahead: ${run.horizon} | Status: ${escapeHtml(run.status)}</div>
            </article>
          `
        )
        .join("")
    : '<article class="list-item">No saved future estimates yet.</article>';
}

function renderClassificationRuns() {
  const currentRunId = Number(elements.classificationRunSelect.value);
  elements.classificationRunSelect.innerHTML = state.classificationRuns.length
    ? state.classificationRuns
        .map(
          (run) =>
            `<option value="${run.id}">Saved recommendation ${run.id} - ${escapeHtml(run.target_column)}</option>`
        )
        .join("")
    : '<option value="">Create a recommendation model first</option>';

  if (state.classificationRuns.length) {
    const selectedRun =
      state.classificationRuns.find((run) => run.id === currentRunId) || state.classificationRuns[0];
    elements.classificationRunSelect.value = String(selectedRun.id);
    renderClassificationFeatureFields(selectedRun);
  } else {
    renderClassificationFeatureFields(null);
  }

  elements.classificationRunLibrary.innerHTML = state.classificationRuns.length
    ? state.classificationRuns
        .map(
          (run) => `
            <article class="list-item">
              <strong>Saved recommendation ${run.id}</strong>
              <div class="list-meta">From: ${escapeHtml(getDatasetName(run.dataset_id))}</div>
              <div class="list-meta">Answer column: ${escapeHtml(run.target_column)} | Status: ${escapeHtml(run.status)}</div>
            </article>
          `
        )
        .join("")
    : '<article class="list-item">No saved recommendation models yet.</article>';
}

function renderDocuments(documents) {
  elements.documentList.innerHTML = documents.length
    ? documents
        .map(
          (document) => `
            <article class="list-item">
              <strong>${escapeHtml(document.title)}</strong>
              <div class="list-meta">${escapeHtml(document.filename)}</div>
              <div class="list-meta">${document.chunk_count} sections ready to use</div>
            </article>
          `
        )
        .join("")
    : '<article class="list-item">No reports uploaded yet.</article>';
}

function renderForecastMetrics(run = null) {
  const evaluation = run?.metrics_json?.evaluation || {};
  const items = [
    ["Average miss", evaluation.mae],
    ["Typical miss", evaluation.rmse],
    ["Fit score", evaluation.r2],
    ["Percent miss", evaluation.mape],
  ];
  elements.forecastMetrics.innerHTML = items
    .map(
      ([label, value]) => `
        <article class="metric-card">
          <span>${label}</span>
          <strong>${value !== undefined ? formatNumber(value) : "-"}</strong>
        </article>
      `
    )
    .join("");
}

function renderClassificationMetrics(run = null) {
  const evaluation = run?.metrics_json?.evaluation || {};
  const items = [
    ["Right answers", evaluation.accuracy],
    ["Clean picks", evaluation.precision_macro],
    ["How much it catches", evaluation.recall_macro],
    ["Overall balance", evaluation.f1_macro],
  ];
  elements.classificationMetrics.innerHTML = items
    .map(
      ([label, value]) => `
        <article class="metric-card">
          <span>${label}</span>
          <strong>${value !== undefined ? formatNumber(value) : "-"}</strong>
        </article>
      `
    )
    .join("");
}

function renderInsights(insights = {}, warnings = []) {
  const globalItems = insights.global_top_features || [];
  const localItems = insights.local_top_contributors || [];
  const warningBlock = warnings.length
    ? `
      <article class="list-item">
        <strong>Things to know</strong>
        <div class="pill-list">
          ${warnings.map((warning) => `<span class="pill neutral">${escapeHtml(warning)}</span>`).join("")}
        </div>
      </article>
    `
    : "";
  elements.shapInsights.innerHTML = `
    ${warningBlock}
    <article class="list-item">
      <strong>What mattered most overall</strong>
      <div class="pill-list">
        ${globalItems
          .map((item) => `<span class="pill">${escapeHtml(item.feature)}: ${formatNumber(item.importance)}</span>`)
          .join("") || "<span class='pill'>No explanation summary yet</span>"}
      </div>
    </article>
    <article class="list-item">
      <strong>What mattered most for the latest result</strong>
      <div class="pill-list">
        ${localItems
          .map((item) => `<span class="pill">${escapeHtml(item.feature)}: ${formatNumber(item.contribution)}</span>`)
          .join("") || "<span class='pill'>No explanation summary yet</span>"}
      </div>
    </article>
  `;
}

function renderClassificationInsights(run = null, insights = {}) {
  const topFeatures = insights.top_features || run?.metrics_json?.insights?.top_features || [];
  const classes = run?.metrics_json?.classes || [];
  const holdoutPredictions = run?.metrics_json?.holdout_predictions || [];
  elements.classificationInsights.innerHTML = `
    <article class="list-item">
      <strong>What mattered most</strong>
      <div class="pill-list">
        ${topFeatures
          .map((item) => `<span class="pill">${escapeHtml(item.feature)}: ${formatNumber(item.importance)}</span>`)
          .join("") || "<span class='pill'>No importance summary yet</span>"}
      </div>
    </article>
    <article class="list-item">
      <strong>Possible answers</strong>
      <div class="pill-list">
        ${classes.map((item) => `<span class="pill neutral">${escapeHtml(item)}</span>`).join("") || "<span class='pill neutral'>No answer list yet</span>"}
      </div>
    </article>
    <article class="list-item">
      <strong>Example checks</strong>
      ${
        holdoutPredictions.length
          ? holdoutPredictions
              .slice(0, 4)
              .map((item) => `<div class="list-meta">Expected: ${escapeHtml(item.actual)} | Predicted: ${escapeHtml(item.predicted)}</div>`)
              .join("")
          : "<div class='list-meta'>No example checks loaded yet.</div>"
      }
    </article>
  `;
}

function renderCharts(charts = {}) {
  setImageSource(elements.forecastChart, charts.forecast_chart || "");
  setImageSource(elements.globalShapChart, charts.global_importance || "");
  setImageSource(elements.localShapChart, charts.local_explanation || "");
}

function renderClassificationCharts(charts = {}) {
  setImageSource(elements.classificationImportanceChart, charts.importance_chart || "");
  setImageSource(elements.classificationConfusionChart, charts.confusion_chart || "");
}

function renderForecastProjection(rows = []) {
  elements.forecastProjection.innerHTML = rows.length
    ? rows
        .slice(0, 8)
        .map(
          (row) => `
            <article class="list-item">
              <strong>${escapeHtml(row.timestamp)}</strong>
              <div class="list-meta">Estimated value: ${formatNumber(row.prediction)}</div>
            </article>
          `
        )
        .join("")
    : '<article class="list-item">No future values loaded yet.</article>';
}

function renderClassificationFeatureFields(run = null) {
  const featureColumns = run?.metrics_json?.feature_columns || [];
  const featureHints = run?.metrics_json?.feature_hints || {};
  const numericColumns = new Set(run?.metrics_json?.numeric_columns || []);
  if (!featureColumns.length) {
    elements.classificationFeatureFields.innerHTML =
      '<article class="list-item">Create or open a saved recommendation to enter values here.</article>';
    return;
  }

  elements.classificationFeatureFields.innerHTML = featureColumns
    .map((feature) => {
      const hint = featureHints[feature] || {};
      const isNumeric = numericColumns.has(feature);
      const defaultValue = hint.default === undefined || hint.default === null ? "" : escapeHtml(String(hint.default));
      const placeholder =
        hint.example === undefined || hint.example === null
          ? isNumeric
            ? "Enter a number"
            : "Enter a label"
          : `Example: ${hint.example}`;
      const knownValues =
        Array.isArray(hint.known_values) && hint.known_values.length
          ? `<small class="list-meta">Common values: ${escapeHtml(hint.known_values.join(", "))}</small>`
          : "";
      return `
        <label>
          <span>${escapeHtml(feature)}</span>
          <input
            type="${isNumeric ? "number" : "text"}"
            name="${escapeHtml(feature)}"
            ${isNumeric ? 'step="any" inputmode="decimal"' : ""}
            placeholder="${escapeHtml(placeholder)}"
            value="${defaultValue}"
            required
          />
          ${knownValues}
        </label>
      `;
    })
    .join("");
}

function renderClassificationPrediction(prediction = null) {
  if (!prediction) {
    elements.classificationPrediction.innerHTML =
      '<article class="list-item">No recommendation yet. Open a saved model and fill in the values above.</article>';
    return;
  }

  elements.classificationPrediction.innerHTML = `
    <article class="list-item">
      <strong>Best match</strong>
      <div class="list-meta">${escapeHtml(prediction.predicted_label)}</div>
    </article>
    <article class="list-item">
      <strong>Other likely matches</strong>
      <div class="pill-list">
        ${(prediction.probabilities || [])
          .map((item) => `<span class="pill">${escapeHtml(item.label)}: ${formatNumber(item.probability)}</span>`)
          .join("")}
      </div>
    </article>
  `;
}

function renderAssistantContext() {
  if (!state.token) {
    elements.assistantContextChips.innerHTML = "";
    return;
  }

  const selectedDataset = getDatasetById(getActiveAssistantDatasetId());
  const activeModeLabel = state.resultsMode === "classification" ? "Best match" : "Future values";
  const activeModelLabel =
    state.resultsMode === "classification"
      ? getActiveAssistantClassificationRunId()
        ? `Saved recommendation ${getActiveAssistantClassificationRunId()}`
        : "No saved recommendation yet"
      : getActiveAssistantForecastRunId()
        ? `Saved estimate ${getActiveAssistantForecastRunId()}`
        : "No saved estimate yet";

  const chips = [
    { label: "Answer mode", value: state.assistantMode === "rag" ? "Reports only" : "Smart answer" },
    { label: "File", value: selectedDataset?.name || "Not chosen" },
    { label: "Using", value: activeModeLabel },
    { label: "Model", value: activeModelLabel },
    { label: "Reports", value: `${state.documents.length}` },
  ];

  elements.assistantContextChips.innerHTML = chips
    .map((chip) => `<span class="pill neutral">${escapeHtml(chip.label)}: ${escapeHtml(chip.value)}</span>`)
    .join("");
}

function renderAssistantSources(sources = []) {
  state.lastAssistantSources = sources;
  elements.assistantSourceList.innerHTML = sources.length
    ? sources
        .map(
          (source) => `
            <article class="list-item">
              <strong>${escapeHtml(source.document_title)}</strong>
              <div class="list-meta">${escapeHtml(source.excerpt)}</div>
            </article>
          `
        )
        .join("")
    : '<article class="list-item">No report evidence used yet.</article>';
}

function appendMessage(role, text) {
  const container = document.createElement("article");
  container.className = `message ${role}`;
  container.textContent = text;
  elements.conversation.appendChild(container);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  return container;
}

function setAssistantPending(isPending) {
  state.assistantPending = isPending;
  elements.assistantSubmit.disabled = isPending;
  elements.assistantSubmit.textContent = isPending ? "Working on it..." : "Ask";
  const textarea = elements.assistantForm.querySelector("textarea[name='question']");
  if (textarea) {
    textarea.disabled = isPending;
  }
}

async function loadDashboard() {
  if (!state.token) {
    renderMetrics();
    renderOverviewSummary(null);
    renderHostingSummary(null);
    return;
  }

  state.dashboardSummary = await apiFetch("/api/v1/metrics/summary");
  renderMetrics(state.dashboardSummary.totals);
  renderOverviewSummary(state.dashboardSummary);
  renderHostingSummary(state.dashboardSummary.hosting_profile);
}

async function loadDatasets() {
  if (!state.token) {
    state.datasets = [];
    renderDatasets();
    return;
  }
  state.datasets = await apiFetch("/api/v1/data/datasets");
  renderDatasets();
}

async function loadDocuments() {
  if (!state.token) {
    state.documents = [];
    renderDocuments([]);
    return;
  }
  state.documents = await apiFetch("/api/v1/data/documents");
  renderDocuments(state.documents);
  renderAssistantContext();
}

async function loadRuns() {
  if (!state.token) {
    state.runs = [];
    renderRuns();
    renderForecastMetrics();
    renderForecastProjection([]);
    renderCharts({});
    renderInsights({});
    return;
  }
  state.runs = await apiFetch("/api/v1/forecast/runs");
  renderRuns();
  renderTrainRunSummary();
  renderAssistantContext();
}

async function loadClassificationRuns() {
  if (!state.token) {
    state.classificationRuns = [];
    renderClassificationRuns();
    renderClassificationMetrics();
    renderClassificationCharts({});
    renderClassificationInsights();
    renderClassificationPrediction();
    return;
  }
  state.classificationRuns = await apiFetch("/api/v1/classification/runs");
  renderClassificationRuns();
  renderTrainRunSummary();
  renderAssistantContext();
}

async function loadRecentTasks() {
  if (!state.token) {
    state.recentTasks = [];
    renderRecentTasks();
    return;
  }
  state.recentTasks = await apiFetch("/api/v1/tasks/recent");
  renderRecentTasks();
}

async function bootstrapWorkspace() {
  if (!state.token) {
    renderMetrics();
    renderOverviewSummary(null);
    renderHostingSummary(null);
    renderRecentTasks();
    renderDatasets();
    renderDocuments([]);
    renderRuns();
    renderForecastMetrics();
    renderCharts({});
    renderInsights({});
    renderForecastProjection([]);
    renderClassificationRuns();
    renderClassificationMetrics();
    renderClassificationCharts({});
    renderClassificationInsights();
    renderClassificationPrediction();
    renderAssistantContext();
    renderAssistantSources([]);
    return;
  }

  await Promise.all([
    loadDashboard(),
    loadDatasets(),
    loadDocuments(),
    loadRuns(),
    loadClassificationRuns(),
    loadRecentTasks(),
  ]);
}

async function handleAuth(event) {
  event.preventDefault();
  const form = new FormData(elements.authForm);

  try {
    let payload;
    if (state.authMode === "register") {
      payload = await apiFetch("/api/v1/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: form.get("email"),
          password: form.get("password"),
          full_name: form.get("full_name") || null,
        }),
      });
    } else {
      const loginBody = new URLSearchParams();
      loginBody.set("username", form.get("email"));
      loginBody.set("password", form.get("password"));
      payload = await apiFetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: loginBody.toString(),
      });
    }

    state.token = payload.access_token;
    state.user = payload.user;
    localStorage.setItem("decision-intel-token", state.token);
    localStorage.setItem("decision-intel-user", JSON.stringify(state.user));
    setSession();
    setStatus(elements.authStatus, `You are signed in as ${payload.user.email}.`);
    elements.authForm.reset();
    await bootstrapWorkspace();
    setActiveView("overview");
  } catch (error) {
    setStatus(elements.authStatus, error.message, true);
  }
}

function logout() {
  state.token = "";
  state.user = null;
  state.datasets = [];
  state.documents = [];
  state.runs = [];
  state.classificationRuns = [];
  state.recentTasks = [];
  localStorage.removeItem("decision-intel-token");
  localStorage.removeItem("decision-intel-user");
  rememberLastSaved("Nothing saved yet");
  setSession();
  renderMetrics();
  renderOverviewSummary(null);
  renderHostingSummary(null);
  renderRecentTasks();
  renderDatasetPreview([]);
  renderDocuments([]);
  renderRuns();
  renderForecastMetrics();
  renderForecastProjection([]);
  renderCharts({});
  renderInsights({});
  renderClassificationRuns();
  renderClassificationMetrics();
  renderClassificationCharts({});
  renderClassificationInsights();
  renderClassificationPrediction();
  renderAssistantContext();
  renderAssistantSources([]);
  setActiveView("auth");
}

async function handleCsvFileSelection(event) {
  const file = event.target.files?.[0];
  if (!file) {
    state.uploadHeaders = [];
    populateSelect(elements.uploadTargetSelect, [], { blankLabel: "Let the app suggest one" });
    populateSelect(elements.uploadTimeSelect, [], { blankLabel: "No date column" });
    elements.uploadCsvHelper.textContent = "Pick a CSV file to load its columns here before you upload.";
    return;
  }

  try {
    const headers = await loadCsvHeaders(file);
    state.uploadHeaders = headers;
    populateSelect(elements.uploadTargetSelect, headers, {
      blankLabel: "Let the app suggest one",
      preferredValue: inferLikelyTarget(headers),
    });
    populateSelect(elements.uploadTimeSelect, headers, {
      blankLabel: "No date column",
      preferredValue: inferLikelyTime(headers),
    });
    elements.uploadCsvHelper.textContent = headers.length
      ? `Loaded columns: ${headers.join(", ")}`
      : "I could not read the column names from that file.";
  } catch (error) {
    elements.uploadCsvHelper.textContent = `I could not read that file yet. ${error.message}`;
  }
}

async function handleCsvUpload(event) {
  event.preventDefault();
  const formData = new FormData(elements.csvUploadForm);
  try {
    setStatus(elements.csvStatus, "Uploading your file...");
    const response = await apiFetch("/api/v1/data/upload/csv", {
      method: "POST",
      body: formData,
    });
    const warnings = response.warnings || [];
    const workflow = response.dataset.metadata_json?.recommended_workflow || "exploration";
    const workflowLabel = getWorkflowLabel(workflow);
    const summary = response.dataset.metadata_json?.workflow_summary || {};
    const baseMessage = `${summary.headline || `Your file "${response.dataset.name}" is ready.`} Best next page: ${workflowLabel}.`;
    setStatus(elements.csvStatus, warnings.length ? `${baseMessage} ${warnings.join(" ")}` : baseMessage);
    rememberLastSaved(`Saved ${response.dataset.name}`);
    renderDatasetPreview(response.preview_rows);
    await Promise.all([loadDashboard(), loadDatasets(), loadRuns(), loadClassificationRuns(), loadRecentTasks()]);
    state.trainMode = workflow === "classification" ? "classification" : "forecast";
    toggleTrainMode(state.trainMode);
    state.resultsMode = workflow === "classification" ? "classification" : "forecast";
    toggleResultsMode(state.resultsMode);
    setActiveView("forecast");
  } catch (error) {
    setStatus(elements.csvStatus, error.message, true);
  }
}

async function handleDocumentUpload(event) {
  event.preventDefault();
  const formData = new FormData(elements.docUploadForm);
  try {
    setStatus(elements.docStatus, "Uploading your report...");
    const response = await apiFetch("/api/v1/data/upload/document", {
      method: "POST",
      body: formData,
    });
    setStatus(
      elements.docStatus,
      `Your report "${response.document.title}" is ready. I saved ${response.document.chunk_count} readable sections for answers.`
    );
    rememberLastSaved(`Saved ${response.document.title}`);
    await Promise.all([loadDashboard(), loadDocuments(), loadRecentTasks()]);
    setActiveView("assistant");
  } catch (error) {
    setStatus(elements.docStatus, error.message, true);
  }
}

async function handleForecastTrain(event) {
  event.preventDefault();
  const form = new FormData(elements.forecastForm);
  const targetColumn = String(form.get("target_column") || "").trim();

  if (!targetColumn) {
    setStatus(elements.forecastStatus, "Choose the number you want to estimate first.", true);
    return;
  }

  const payload = {
    dataset_id: Number(form.get("dataset_id")),
    target_column: targetColumn,
    time_column: form.get("time_column") || null,
    horizon: Number(form.get("horizon")),
    test_size: Number(form.get("test_size")),
    lags: String(form.get("lags"))
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => !Number.isNaN(value)),
  };

  try {
    setStatus(elements.forecastStatus, "Creating your future estimate...");
    const response = await apiFetch("/api/v1/forecast/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const warnings = response.run.metrics_json?.warnings || [];
    setStatus(
      elements.forecastStatus,
      warnings.length ? `Your future estimate is ready. ${warnings.join(" ")}` : "Your future estimate is ready."
    );
    rememberLastSaved(`Saved future estimate ${response.run.id}`);
    renderForecastMetrics(response.run);
    renderCharts(response.charts);
    renderInsights(response.insights, warnings);
    renderForecastProjection(response.run.metrics_json?.future_forecast || []);
    await Promise.all([loadDashboard(), loadRuns(), loadRecentTasks()]);
    elements.runSelect.value = String(response.run.id);
    toggleResultsMode("forecast");
    toggleResultsTab("summary");
    setActiveView("classification");
  } catch (error) {
    setStatus(elements.forecastStatus, error.message, true);
  }
}

async function loadSelectedRun() {
  const runId = elements.runSelect.value;
  if (!runId) {
    return;
  }

  try {
    const response = await apiFetch(`/api/v1/forecast/runs/${runId}`);
    renderForecastMetrics(response.run);
    renderCharts(response.charts);
    renderInsights(response.insights, response.run.metrics_json?.warnings || []);
    renderForecastProjection(response.run.metrics_json?.future_forecast || []);
    toggleResultsMode("forecast");
    setStatus(elements.forecastStatus, "Saved future estimate loaded.");
    setActiveView("classification");
  } catch (error) {
    setStatus(elements.forecastStatus, error.message, true);
  }
}

async function predictAgain() {
  const runId = elements.runSelect.value;
  if (!runId) {
    setStatus(elements.forecastStatus, "Choose a saved future estimate first.", true);
    return;
  }

  try {
    const payload = await apiFetch(`/api/v1/forecast/runs/${runId}/predict?horizon=14`, {
      method: "POST",
    });
    renderForecastProjection(payload.forecast || []);
    setStatus(elements.forecastStatus, `Created ${payload.horizon} new future values.`);
  } catch (error) {
    setStatus(elements.forecastStatus, error.message, true);
  }
}

async function handleClassificationTrain(event) {
  event.preventDefault();
  const form = new FormData(elements.classificationForm);
  const payload = {
    dataset_id: Number(form.get("dataset_id")),
    target_column: String(form.get("target_column") || "").trim(),
    test_size: Number(form.get("test_size")),
    max_classes_for_report: Number(form.get("max_classes_for_report")),
  };

  if (!payload.target_column) {
    setStatus(elements.classificationStatus, "Choose the answer column you want the app to learn.", true);
    return;
  }

  try {
    setStatus(elements.classificationStatus, "Creating your recommendation model...");
    const response = await apiFetch("/api/v1/classification/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const classCount = response.run.metrics_json?.classes?.length || 0;
    setStatus(
      elements.classificationStatus,
      `Your recommendation model is ready. It learned ${classCount} possible answers.`
    );
    rememberLastSaved(`Saved recommendation ${response.run.id}`);
    renderClassificationMetrics(response.run);
    renderClassificationCharts(response.charts);
    renderClassificationInsights(response.run, response.insights);
    renderClassificationFeatureFields(response.run);
    renderClassificationPrediction();
    await Promise.all([loadDashboard(), loadClassificationRuns(), loadRecentTasks()]);
    elements.classificationRunSelect.value = String(response.run.id);
    toggleResultsMode("classification");
    toggleResultsTab("summary");
    setActiveView("classification");
  } catch (error) {
    setStatus(elements.classificationStatus, error.message, true);
  }
}

async function loadSelectedClassificationRun() {
  const runId = elements.classificationRunSelect.value;
  if (!runId) {
    return;
  }

  try {
    const response = await apiFetch(`/api/v1/classification/runs/${runId}`);
    renderClassificationMetrics(response.run);
    renderClassificationCharts(response.charts);
    renderClassificationInsights(response.run, response.insights);
    renderClassificationFeatureFields(response.run);
    renderClassificationPrediction();
    toggleResultsMode("classification");
    setStatus(elements.classificationStatus, "Saved recommendation loaded.");
    setActiveView("classification");
  } catch (error) {
    setStatus(elements.classificationStatus, error.message, true);
  }
}

async function handleClassificationPredict(event) {
  event.preventDefault();
  const runId = elements.classificationRunSelect.value;
  if (!runId) {
    setStatus(elements.classificationStatus, "Choose a saved recommendation first.", true);
    return;
  }

  const run = getClassificationRunById(runId);
  const numericColumns = new Set(run?.metrics_json?.numeric_columns || []);
  const inputs = [...elements.classificationFeatureFields.querySelectorAll("input")];
  if (!inputs.length) {
    setStatus(elements.classificationStatus, "This saved recommendation is missing input fields right now.", true);
    return;
  }

  const missingFields = inputs
    .filter((input) => !String(input.value || "").trim())
    .map((input) => input.name);
  if (missingFields.length) {
    setStatus(elements.classificationStatus, `Please fill in: ${missingFields.join(", ")}`, true);
    return;
  }

  const featureValues = {};
  for (const input of inputs) {
    const rawValue = String(input.value || "").trim();
    featureValues[input.name] = numericColumns.has(input.name) ? Number(rawValue) : rawValue;
  }

  try {
    const response = await apiFetch(`/api/v1/classification/runs/${runId}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature_values: featureValues }),
    });
    renderClassificationPrediction(response);
    setStatus(elements.classificationStatus, "Your recommendation is ready.");
  } catch (error) {
    setStatus(elements.classificationStatus, error.message, true);
  }
}

async function handleAssistant(event) {
  event.preventDefault();
  if (state.assistantPending) {
    return;
  }

  const form = new FormData(elements.assistantForm);
  const question = String(form.get("question") || "").trim();
  if (!question) {
    return;
  }

  appendMessage("user", question);
  const placeholder = appendMessage("assistant", "Working on it...");
  elements.assistantForm.reset();
  setAssistantPending(true);

  try {
    if (state.assistantMode === "rag") {
      const response = await apiFetch("/api/v1/rag/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 4 }),
      });
      placeholder.textContent = response.answer;
      renderAssistantSources(response.sources || []);
      return;
    }

    const response = await apiFetch("/api/v1/assistant/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        dataset_id: getActiveAssistantDatasetId(),
        forecast_run_id: getActiveAssistantForecastRunId(),
        classification_run_id: getActiveAssistantClassificationRunId(),
        top_k: 4,
      }),
    });
    placeholder.textContent = response.answer;
    renderAssistantSources(response.sources || []);
  } catch (error) {
    placeholder.textContent = `Sorry, I could not answer that yet. ${error.message}`;
  } finally {
    setAssistantPending(false);
  }
}

elements.pageNav.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-view]");
  if (!button || button.disabled) {
    return;
  }
  setActiveView(button.dataset.view);
});

elements.authMode.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button) {
    return;
  }
  toggleAuthMode(button.dataset.mode);
});

elements.assistantMode.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button) {
    return;
  }
  toggleAssistantMode(button.dataset.mode);
});

elements.trainMode.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button) {
    return;
  }
  toggleTrainMode(button.dataset.mode);
});

elements.resultsMode.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (!button) {
    return;
  }
  toggleResultsMode(button.dataset.mode);
});

elements.resultsTab.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-tab]");
  if (!button) {
    return;
  }
  toggleResultsTab(button.dataset.tab);
});

elements.logoutBtn.addEventListener("click", logout);
elements.authContinueBtn.addEventListener("click", () => setActiveView("overview"));
elements.authForm.addEventListener("submit", handleAuth);
elements.csvFileInput.addEventListener("change", handleCsvFileSelection);
elements.csvUploadForm.addEventListener("submit", handleCsvUpload);
elements.docUploadForm.addEventListener("submit", handleDocumentUpload);
elements.forecastForm.addEventListener("submit", handleForecastTrain);
elements.refreshRun.addEventListener("click", loadSelectedRun);
elements.predictAgain.addEventListener("click", predictAgain);
elements.classificationForm.addEventListener("submit", handleClassificationTrain);
elements.refreshClassificationRun.addEventListener("click", loadSelectedClassificationRun);
elements.classificationPredictForm.addEventListener("submit", handleClassificationPredict);
elements.assistantForm.addEventListener("submit", handleAssistant);
elements.refreshDashboard.addEventListener("click", bootstrapWorkspace);
elements.datasetSelect.addEventListener("change", syncForecastDefaults);
elements.classificationDatasetSelect.addEventListener("change", syncClassificationDefaults);
elements.classificationRunSelect.addEventListener("change", () => {
  renderClassificationFeatureFields(getClassificationRunById(elements.classificationRunSelect.value));
  renderClassificationPrediction();
  renderAssistantContext();
});
elements.runSelect.addEventListener("change", renderAssistantContext);

if (state.token && state.activeView === "auth") {
  state.activeView = "overview";
}

setSession();
toggleAuthMode(state.authMode);
toggleAssistantMode(state.assistantMode);
toggleTrainMode(state.trainMode);
toggleResultsMode(state.resultsMode);
toggleResultsTab(state.resultsTab);
rememberLastSaved(state.lastSavedMessage);
renderMetrics();
renderOverviewSummary(null);
renderHostingSummary(null);
renderRecentTasks();
renderDatasetPreview([]);
renderWorkflowGuide();
renderDocuments([]);
renderRuns();
renderForecastMetrics();
renderCharts({});
renderInsights({});
renderForecastProjection([]);
renderClassificationRuns();
renderClassificationMetrics();
renderClassificationCharts({});
renderClassificationInsights();
renderClassificationPrediction();
renderDatasets();
renderAssistantContext();
renderAssistantSources([]);
setActiveView(state.activeView);
bootstrapWorkspace().catch((error) => {
  appendMessage("assistant", `I could not load everything yet. ${error.message}`);
});
