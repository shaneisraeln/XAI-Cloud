// script.js — XAI Cloud Service Dashboard
// Connects frontend UI to FastAPI backend at http://localhost:8000

const API_BASE = "http://localhost:8000";

// ── Feature names (matching Breast Cancer Wisconsin dataset schema) ───────────
const FEATURE_NAMES = [
  "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
  "mean compactness","mean concavity","mean concave points","mean symmetry",
  "mean fractal dimension","radius error","texture error","perimeter error",
  "area error","smoothness error","compactness error","concavity error",
  "concave points error","symmetry error","fractal dimension error",
  "worst radius","worst texture","worst perimeter","worst area",
  "worst smoothness","worst compactness","worst concavity",
  "worst concave points","worst symmetry","worst fractal dimension"
];

// ── Predefined dataset samples ────────────────────────────────────────────────
const SAMPLES = {
  malignant: [
    17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
    1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
    25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
  ],
  benign: [
    13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,
    0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,
    15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
  ]
};

// ── Feature ranges [min, max] for realistic input randomisation ─────────────
const FEATURE_RANGES = [
  [6.98, 28.11],    // mean radius
  [9.71, 39.28],    // mean texture
  [43.79, 188.5],   // mean perimeter
  [143.5, 2501.0],  // mean area
  [0.052, 0.163],   // mean smoothness
  [0.019, 0.345],   // mean compactness
  [0.0, 0.427],     // mean concavity
  [0.0, 0.201],     // mean concave points
  [0.106, 0.304],   // mean symmetry
  [0.05, 0.097],    // mean fractal dimension
  [0.111, 2.873],   // radius error
  [0.36, 4.885],    // texture error
  [0.757, 21.98],   // perimeter error
  [6.8, 542.2],     // area error
  [0.0017, 0.031],  // smoothness error
  [0.0022, 0.135],  // compactness error
  [0.0, 0.396],     // concavity error
  [0.0, 0.052],     // concave points error
  [0.0078, 0.078],  // symmetry error
  [0.0008, 0.03],   // fractal dimension error
  [7.93, 36.04],    // worst radius
  [12.02, 49.54],   // worst texture
  [50.41, 251.2],   // worst perimeter
  [185.2, 4254.0],  // worst area
  [0.071, 0.223],   // worst smoothness
  [0.027, 1.058],   // worst compactness
  [0.0, 1.252],     // worst concavity
  [0.0, 0.291],     // worst concave points
  [0.156, 0.664],   // worst symmetry
  [0.055, 0.207]    // worst fractal dimension
];

// ── Baseline latency tracker ─────────────────────────────────────────────────
let baselineMs = null;

// ── Build feature input grid ──────────────────────────────────────────────────
function buildFeatureGrid() {
  const grid = document.getElementById("feature-grid");
  grid.innerHTML = "";
  FEATURE_NAMES.forEach((name, i) => {
    const wrap = document.createElement("div");
    wrap.className = "feature-input-wrap";
    wrap.innerHTML = `
      <label for="f${i}" title="${name}">${name}</label>
      <input type="number" step="any" id="f${i}" value="${SAMPLES.malignant[i]}" />
    `;
    grid.appendChild(wrap);
  });
}

// ── Load predefined sample ───────────────────────────────────────────────────
function loadSample() {
  const key = document.getElementById("sample-select").value;
  const vals = SAMPLES[key];
  FEATURE_NAMES.forEach((_, i) => {
    document.getElementById(`f${i}`).value = vals[i];
  });
}

// ── Randomize input values dynamically ───────────────────────────────────────
function randomizeInputs() {
  FEATURE_NAMES.forEach((_, i) => {
    const [min, max] = FEATURE_RANGES[i];
    const val = min + Math.random() * (max - min);
    const formatted = val > 10 ? val.toFixed(2) : (val > 1 ? val.toFixed(3) : val.toFixed(4));
    document.getElementById(`f${i}`).value = formatted;
  });
}

// ── Read current feature values ──────────────────────────────────────────────
function getFeatures() {
  return FEATURE_NAMES.map((_, i) => {
    const raw = parseFloat(document.getElementById(`f${i}`).value);
    return isNaN(raw) ? 0.0 : raw;
  });
}

// ── Show / hide loading indicator ─────────────────────────────────────────────
function setLoading(visible, text = "Computing predictions and explanations...") {
  const el = document.getElementById("loading");
  document.getElementById("loading-text").textContent = text;
  el.classList.toggle("hidden", !visible);
}

// ── Show a section card ───────────────────────────────────────────────────────
function showSection(id) {
  document.getElementById(id).classList.remove("hidden");
}

// ── Render prediction result ──────────────────────────────────────────────────
function renderPrediction(prediction, probability, inferenceMs) {
  showSection("section-prediction");
  const label = document.getElementById("prediction-label");
  label.textContent = prediction.toUpperCase();
  label.className = "prediction-label " + prediction.toLowerCase();

  document.getElementById("pred-confidence").textContent =
    (probability * 100).toFixed(1) + "%";
  document.getElementById("pred-inference-time").textContent =
    inferenceMs.toFixed(2) + " ms";

  const bar = document.getElementById("confidence-bar");
  bar.style.width = (probability * 100) + "%";
  bar.style.background = prediction === "malignant" ? "var(--danger)" : "var(--success)";
}

// ── Render horizontal bar chart (SHAP or LIME) ──────────────────────────────
function renderBarChart(containerId, labels, values, maxItems = 10) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  const items = labels.slice(0, maxItems).map((l, i) => ({ label: l, value: values[i] }));
  const maxAbs = Math.max(...items.map(d => Math.abs(d.value)), 0.0001);

  items.forEach(({ label, value }) => {
    const pct = (Math.abs(value) / maxAbs) * 48; // max 48% of half bar width
    const row = document.createElement("div");
    row.className = "chart-row";

    const nameEl = document.createElement("div");
    nameEl.className = "chart-feature-name";
    nameEl.textContent = label;
    nameEl.title = label;

    const barWrap = document.createElement("div");
    barWrap.className = "chart-bar-wrap";

    const bar = document.createElement("div");
    bar.className = "chart-bar " + (value >= 0 ? "bar-positive" : "bar-negative");
    bar.style.width = pct + "%";
    barWrap.appendChild(bar);

    const valEl = document.createElement("div");
    valEl.className = "chart-value";
    valEl.textContent = (value >= 0 ? "+" : "") + value.toFixed(4);

    row.appendChild(nameEl);
    row.appendChild(barWrap);
    row.appendChild(valEl);
    container.appendChild(row);
  });
}

// ── Render timing chips below chart ──────────────────────────────────────────
function renderTiming(containerId, chips) {
  const el = document.getElementById(containerId);
  el.innerHTML = chips.map(
    c => `<div class="timing-chip">${c.label}: <strong>${c.value}</strong></div>`
  ).join("");
}

// ── Update performance panel ──────────────────────────────────────────────────
function updatePerformancePanel(inferMs, shapMs, limeMs, totalMs, overheadMs, overheadPct) {
  showSection("section-performance");
  const fmt = v => v != null ? v.toFixed(2) + " ms" : "-";
  document.getElementById("perf-inference").textContent = fmt(inferMs);
  document.getElementById("perf-shap").textContent      = shapMs != null ? fmt(shapMs) : "-";
  document.getElementById("perf-lime").textContent      = limeMs != null ? fmt(limeMs) : "-";
  document.getElementById("perf-total").textContent     = fmt(totalMs);
  document.getElementById("perf-overhead-ms").textContent  = overheadMs != null ? fmt(overheadMs) : "-";
  document.getElementById("perf-overhead-pct").textContent = overheadPct != null
    ? overheadPct.toFixed(1) + "%" : "-";
}

// ── Update research table ─────────────────────────────────────────────────────
function updateResearchTable(config, ms) {
  const ids = {
    baseline: "rt-baseline",
    shap:     "rt-shap",
    lime:     "rt-lime",
    combined: "rt-combined",
  };
  const overheadIds = {
    shap:     "rt-shap-overhead",
    lime:     "rt-lime-overhead",
    combined: "rt-combined-overhead",
  };

  if (ids[config]) {
    document.getElementById(ids[config]).textContent = ms.toFixed(2) + " ms";
  }
  if (config === "baseline") {
    baselineMs = ms;
    ["shap","lime","combined"].forEach(k => {
      const el = document.getElementById(overheadIds[k]);
      const valEl = document.getElementById(ids[k]);
      const existing = parseFloat(valEl.textContent);
      if (!isNaN(existing) && baselineMs) {
        const oh = existing - baselineMs;
        const pct = (oh / baselineMs * 100);
        el.textContent = `+${oh.toFixed(1)} ms (${pct.toFixed(0)}%)`;
      }
    });
  } else if (overheadIds[config] && baselineMs) {
    const oh = ms - baselineMs;
    const pct = (oh / baselineMs * 100);
    document.getElementById(overheadIds[config]).textContent =
      `+${oh.toFixed(1)} ms (${pct.toFixed(0)}%)`;
  }
}

// ── API call wrapper ──────────────────────────────────────────────────────────
async function callAPI(endpoint, features) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

// ── POST /predict ─────────────────────────────────────────────────────────────
async function runPredict() {
  setLoading(true, "Running baseline inference...");
  try {
    const features = getFeatures();
    const data = await callAPI("/predict", features);
    renderPrediction(data.prediction, data.probability, data.inference_time_ms);
    updateResearchTable("baseline", data.inference_time_ms);
    updatePerformancePanel(data.inference_time_ms, null, null, data.inference_time_ms, null, null);
  } catch (e) {
    alert("Inference Error: " + e.message);
  } finally {
    setLoading(false);
  }
}

// ── POST /explain/shap ────────────────────────────────────────────────────────
async function runSHAP() {
  setLoading(true, "Computing TreeExplainer SHAP values...");
  try {
    const features = getFeatures();
    const data = await callAPI("/explain/shap", features);

    renderPrediction(data.prediction, data.probability, data.inference_time_ms);
    showSection("section-shap");
    renderBarChart("shap-chart", data.features, data.contributions, 10);
    renderTiming("shap-timing", [
      { label: "Inference", value: data.inference_time_ms.toFixed(2) + " ms" },
      { label: "SHAP",      value: data.explanation_time_ms.toFixed(2) + " ms" },
      { label: "Total",     value: data.total_time_ms.toFixed(2) + " ms" },
    ]);

    updateResearchTable("baseline", data.inference_time_ms);
    updateResearchTable("shap", data.total_time_ms);

    const overhead = data.total_time_ms - data.inference_time_ms;
    const pct = (overhead / data.inference_time_ms) * 100;
    updatePerformancePanel(data.inference_time_ms, data.explanation_time_ms, null,
      data.total_time_ms, overhead, pct);
  } catch (e) {
    alert("SHAP Error: " + e.message);
  } finally {
    setLoading(false);
  }
}

// ── POST /explain/lime ────────────────────────────────────────────────────────
async function runLIME() {
  setLoading(true, "Computing LimeTabularExplainer perturbations...");
  try {
    const features = getFeatures();
    const data = await callAPI("/explain/lime", features);

    renderPrediction(data.prediction, data.probability, data.inference_time_ms);
    showSection("section-lime");
    renderBarChart("lime-chart", data.feature_rules, data.contributions, 10);
    renderTiming("lime-timing", [
      { label: "Inference", value: data.inference_time_ms.toFixed(2) + " ms" },
      { label: "LIME",      value: data.explanation_time_ms.toFixed(2) + " ms" },
      { label: "Total",     value: data.total_time_ms.toFixed(2) + " ms" },
    ]);

    updateResearchTable("baseline", data.inference_time_ms);
    updateResearchTable("lime", data.total_time_ms);

    const overhead = data.total_time_ms - data.inference_time_ms;
    const pct = (overhead / data.inference_time_ms) * 100;
    updatePerformancePanel(data.inference_time_ms, null, data.explanation_time_ms,
      data.total_time_ms, overhead, pct);
  } catch (e) {
    alert("LIME Error: " + e.message);
  } finally {
    setLoading(false);
  }
}

// ── POST /explain (combined) ──────────────────────────────────────────────────
async function runCombined() {
  setLoading(true, "Computing SHAP + LIME explanations...");
  try {
    const features = getFeatures();
    const data = await callAPI("/explain", features);

    renderPrediction(data.prediction, data.probability, data.inference_time_ms);

    showSection("section-shap");
    renderBarChart("shap-chart", data.shap_features, data.shap_contributions, 10);
    renderTiming("shap-timing", [
      { label: "SHAP Time", value: data.shap_explanation_time_ms.toFixed(2) + " ms" },
    ]);

    showSection("section-lime");
    renderBarChart("lime-chart", data.lime_feature_rules, data.lime_contributions, 10);
    renderTiming("lime-timing", [
      { label: "LIME Time", value: data.lime_explanation_time_ms.toFixed(2) + " ms" },
    ]);

    updateResearchTable("baseline", data.inference_time_ms);
    updateResearchTable("combined", data.total_time_ms);

    updatePerformancePanel(
      data.inference_time_ms,
      data.shap_explanation_time_ms,
      data.lime_explanation_time_ms,
      data.total_time_ms,
      data.xai_overhead_ms,
      data.xai_overhead_pct,
    );
  } catch (e) {
    alert("Audit Error: " + e.message);
  } finally {
    setLoading(false);
  }
}

// ── Check API health ─────────────────────────────────────────────────────────
async function checkHealth() {
  const pill = document.getElementById("service-status");
  const textEl = pill.querySelector(".status-text");
  try {
    const r = await fetch(`${API_BASE}/health`);
    const d = await r.json();
    if (d.status === "healthy") {
      if (textEl) textEl.textContent = "API Online";
      pill.className = "status-pill status-ok";
    } else {
      throw new Error("not healthy");
    }
  } catch {
    if (textEl) textEl.textContent = "API Offline";
    pill.className = "status-pill status-error";
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
buildFeatureGrid();
checkHealth();
