const API_URL = "http://localhost:8000";
const $ = (id) => document.getElementById(id);

$("apiUrlLabel").textContent = API_URL;

let trainingChart = null;
let thresholdChart = null;
let currentFeatureCount = null;
let bestThresholdFromTrain = null;

function setStatus(status) {
  const banner = $("statusBanner");
  banner.classList.remove("status-not", "status-training", "status-ready");

  if (status === "READY") {
    banner.classList.add("status-ready");
    banner.textContent = "Status: READY";
    $("predictBtn").disabled = false;
  } else if (status === "TRAINING") {
    banner.classList.add("status-training");
    banner.textContent = "Status: TRAINING...";
    $("predictBtn").disabled = true;
  } else {
    banner.classList.add("status-not");
    banner.textContent = "Status: NOT TRAINED";
    $("predictBtn").disabled = true;
  }
}

function buildPredictInputs(nFeatures) {
  const wrap = $("predictInputs");
  const prev = [];
  for (let i = 0; i < (currentFeatureCount ?? 0); i++) {
    const el = document.getElementById(`feat_${i}`);
    if (el) prev[i] = el.value;
  }

  wrap.innerHTML = "";
  currentFeatureCount = nFeatures;

  if (!nFeatures) return;

  for (let i = 0; i < nFeatures; i++) {
    const label = document.createElement("label");
    label.textContent = `Feature ${i + 1}`;

    const input = document.createElement("input");
    input.type = "number";
    input.step = "1";
    input.id = `feat_${i}`;
    input.value = prev[i] ?? "0";

    label.appendChild(input);
    wrap.appendChild(label);
  }
}

async function getStatus() {
  try {
    const res = await fetch(`${API_URL}/status`);
    const j = await res.json();

    if (j?.data?.status) setStatus(j.data.status);
    if (typeof j?.data?.n_features === "number") buildPredictInputs(j.data.n_features);

    if (j?.data?.persisted) {
      $("modelOut").textContent = `Saved model found: YES\nn_features: ${j.data.n_features ?? "—"}`;
    } else {
      $("modelOut").textContent = `Saved model found: NO\nTrain then click "Save model".`;
    }
  } catch {
    setStatus("NOT TRAINED");
  }
}

function renderTrainingCurve(history) {
  if (!history?.loss?.length) return;

  const labels = history.loss.map((_, i) => String(i + 1));
  const ctx = $("trainingChart").getContext("2d");

  const data = {
    labels,
    datasets: [
      {
        label: "Loss",
        data: history.loss,
        yAxisID: "yLoss",
        borderColor: "#fb7185",
        backgroundColor: "rgba(251,113,133,0.15)",
        tension: 0.25,
        pointRadius: 0,
      },
      {
        label: "Train Acc",
        data: history.train_acc,
        yAxisID: "yAcc",
        borderColor: "#34d399",
        backgroundColor: "rgba(52,211,153,0.12)",
        tension: 0.25,
        pointRadius: 0,
      },
      {
        label: "Test Acc",
        data: history.test_acc,
        yAxisID: "yAcc",
        borderColor: "#60a5fa",
        backgroundColor: "rgba(96,165,250,0.12)",
        tension: 0.25,
        pointRadius: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { labels: { color: "#eaf0ff" } } },
    scales: {
      x: { ticks: { color: "rgba(234,240,255,0.7)" }, grid: { color: "rgba(255,255,255,0.06)" } },
      yLoss: {
        type: "linear",
        position: "left",
        ticks: { color: "rgba(234,240,255,0.7)" },
        grid: { color: "rgba(255,255,255,0.06)" },
        title: { display: true, text: "Loss", color: "rgba(234,240,255,0.85)" },
      },
      yAcc: {
        type: "linear",
        position: "right",
        min: 0,
        max: 1,
        ticks: { color: "rgba(234,240,255,0.7)" },
        grid: { drawOnChartArea: false },
        title: { display: true, text: "Accuracy", color: "rgba(234,240,255,0.85)" },
      },
    },
  };

  if (trainingChart) {
    trainingChart.data = data;
    trainingChart.options = options;
    trainingChart.update();
  } else {
    trainingChart = new Chart(ctx, { type: "line", data, options });
  }

  $("saveChartBtn").disabled = false;
}

function renderThresholdSweep(sweep) {
  if (!sweep?.thresholds?.length) return;

  const ctx = $("thresholdChart").getContext("2d");
  const labels = sweep.thresholds.map((t) => t.toFixed(2));

  const data = {
    labels,
    datasets: [
      {
        label: "Precision",
        data: sweep.precision,
        borderColor: "#a78bfa",
        backgroundColor: "rgba(167,139,250,0.10)",
        tension: 0.25,
        pointRadius: 0,
      },
      {
        label: "Recall",
        data: sweep.recall,
        borderColor: "#34d399",
        backgroundColor: "rgba(52,211,153,0.10)",
        tension: 0.25,
        pointRadius: 0,
      },
      {
        label: "F1",
        data: sweep.f1,
        borderColor: "#60a5fa",
        backgroundColor: "rgba(96,165,250,0.10)",
        tension: 0.25,
        pointRadius: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { labels: { color: "#eaf0ff" } } },
    scales: {
      x: { ticks: { color: "rgba(234,240,255,0.7)" }, grid: { color: "rgba(255,255,255,0.06)" } },
      y: {
        min: 0,
        max: 1,
        ticks: { color: "rgba(234,240,255,0.7)" },
        grid: { color: "rgba(255,255,255,0.06)" },
        title: { display: true, text: "Score", color: "rgba(234,240,255,0.85)" },
      },
    },
  };

  if (thresholdChart) {
    thresholdChart.data = data;
    thresholdChart.options = options;
    thresholdChart.update();
  } else {
    thresholdChart = new Chart(ctx, { type: "line", data, options });
  }

  $("saveThreshChartBtn").disabled = false;
}

function downloadChartPng(chart, filenamePrefix) {
  if (!chart) return;
  const url = chart.toBase64Image("image/png", 1);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filenamePrefix}-${Date.now()}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

$("saveChartBtn").addEventListener("click", () => downloadChartPng(trainingChart, "training-curve"));
$("saveThreshChartBtn").addEventListener("click", () => downloadChartPng(thresholdChart, "threshold-sweep"));

// Threshold sync
$("thresholdRange").addEventListener("input", () => ($("thresholdVal").value = $("thresholdRange").value));
$("thresholdVal").addEventListener("input", () => ($("thresholdRange").value = $("thresholdVal").value));

$("applyBestThresholdBtn").addEventListener("click", () => {
  if (bestThresholdFromTrain == null) return;
  const v = Number(bestThresholdFromTrain).toFixed(2);
  $("thresholdVal").value = v;
  $("thresholdRange").value = v;
});

async function downloadDemo(mode) {
  $("dlOut").textContent = `Downloading ${mode.toUpperCase()} demo CSV...`;
  try {
    const url = `${API_URL}/demo_dataset?mode=${encodeURIComponent(mode)}&rows=3000&anomaly_rate=0.05&seed=7`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const blob = await r.blob();

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cyber_demo_${mode}_3000.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    $("dlOut").textContent = `✅ Downloaded ${a.download}. Upload it in Train section.`;
  } catch (e) {
    $("dlOut").textContent = String(e);
  }
}

$("dlHardBtn").addEventListener("click", () => downloadDemo("hard"));
$("dlEasyBtn").addEventListener("click", () => downloadDemo("easy"));

$("saveModelBtn").addEventListener("click", async () => {
  $("modelOut").textContent = "Saving model...";
  try {
    const res = await fetch(`${API_URL}/save_model`, { method: "POST" });
    $("modelOut").textContent = await res.text();
    await getStatus();
  } catch (e) {
    $("modelOut").textContent = String(e);
  }
});

$("loadModelBtn").addEventListener("click", async () => {
  $("modelOut").textContent = "Loading model...";
  try {
    const res = await fetch(`${API_URL}/load_model`, { method: "POST" });
    $("modelOut").textContent = await res.text();
    await getStatus();
  } catch (e) {
    $("modelOut").textContent = String(e);
  }
});

function renderEvaluation(trainResult) {
  bestThresholdFromTrain = trainResult?.sweep?.best_f1_threshold ?? null;
  $("applyBestThresholdBtn").disabled = bestThresholdFromTrain == null;

  const prev = trainResult?.prevalence_test;
  $("prevalenceLabel").textContent = prev == null ? "Test prevalence: —" : `Test prevalence: ${(prev * 100).toFixed(2)}% anomalies`;

  const cm = trainResult?.eval_best_f1?.confusion;
  if (cm) {
    $("cmTN").textContent = cm.tn;
    $("cmFP").textContent = cm.fp;
    $("cmFN").textContent = cm.fn;
    $("cmTP").textContent = cm.tp;
  }

  const summary = {
    eval_at_05: trainResult?.eval_at_05,
    eval_best_f1: trainResult?.eval_best_f1,
    best_f1_threshold: bestThresholdFromTrain,
    best_f1: trainResult?.sweep?.best_f1,
  };
  $("evalOut").textContent = JSON.stringify(summary, null, 2);

  if (trainResult?.sweep) renderThresholdSweep(trainResult.sweep);
}

$("trainCsvBtn").addEventListener("click", async () => {
  $("trainCsvOut").textContent = "Training from CSV...";
  setStatus("TRAINING");

  try {
    const file = $("csvFile").files?.[0];
    if (!file) throw new Error("Choose a CSV file first.");

    const form = new FormData();
    form.append("file", file);
    form.append("epochs", String(Number($("csvEpochs").value)));
    form.append("lr", String(Number($("csvLr").value)));
    form.append("seed", String(Number($("csvSeed").value)));
    form.append("label_col", $("labelCol").value.trim() || "last");
    form.append("feature_cols", $("featureCols").value.trim());
    form.append("test_ratio", String(Number($("testRatio").value)));
    form.append("eval_every", String(Number($("evalEvery").value)));

    const res = await fetch(`${API_URL}/train_csv`, { method: "POST", body: form });
    const text = await res.text();
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}\n${text}`);

    $("trainCsvOut").textContent = text;

    let parsed = null;
    try {
      parsed = JSON.parse(text);
      if (parsed?.history) renderTrainingCurve(parsed.history);
      renderEvaluation(parsed);
    } catch {}

    await getStatus();

    if (parsed?.sweep?.best_f1_threshold != null) {
      const v = Number(parsed.sweep.best_f1_threshold).toFixed(2);
      $("thresholdVal").value = v;
      $("thresholdRange").value = v;
    }
  } catch (e) {
    $("trainCsvOut").textContent = String(e);
    setStatus("NOT TRAINED");
  }
});

$("predictBtn").addEventListener("click", async () => {
  $("predictOut").textContent = "Scoring...";
  try {
    if (!currentFeatureCount) throw new Error("Train or Load a model first.");

    const features = [];
    for (let i = 0; i < currentFeatureCount; i++) {
      const v = Number(document.getElementById(`feat_${i}`).value);
      if (Number.isNaN(v)) throw new Error(`Feature ${i + 1} is not a number.`);
      features.push(v);
    }

    const threshold = Number($("thresholdVal").value);
    if (Number.isNaN(threshold) || threshold < 0 || threshold > 1) throw new Error("Threshold must be 0..1");

    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features, threshold }),
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}\n${text}`);

    const parsed = JSON.parse(text);
    const tag = parsed.is_anomaly ? "🚨 ANOMALY" : "✅ NORMAL";
    $("predictOut").textContent = `${tag}\n\n` + JSON.stringify(parsed, null, 2);
  } catch (e) {
    $("predictOut").textContent = String(e);
  }
});

// init
getStatus();
