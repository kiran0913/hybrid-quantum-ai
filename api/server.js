import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";

const PORT = process.env.PORT || 8000;
const QUANTUM_URL = process.env.QUANTUM_URL || "http://localhost:8001";

const app = express();
app.use(cors({ origin: "*" })); // dev-only
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

app.get("/health", (_req, res) => res.json({ ok: true, quantum: QUANTUM_URL }));

app.get("/status", async (_req, res) => {
  try {
    const r = await fetch(`${QUANTUM_URL}/status`);
    const data = await r.json();
    res.status(r.ok ? 200 : 502).json({ ok: r.ok, data });
  } catch (e) {
    res.status(502).json({ ok: false, error: String(e) });
  }
});

app.get("/demo_dataset", async (req, res) => {
  try {
    const qs = new URLSearchParams(req.query).toString();
    const url = qs ? `${QUANTUM_URL}/demo_dataset?${qs}` : `${QUANTUM_URL}/demo_dataset`;
    const r = await fetch(url);
    const buf = Buffer.from(await r.arrayBuffer());

    res.status(r.status);
    res.setHeader("Content-Type", r.headers.get("content-type") || "text/csv");
    const cd = r.headers.get("content-disposition");
    if (cd) res.setHeader("Content-Disposition", cd);
    res.send(buf);
  } catch (e) {
    res.status(502).json({ ok: false, error: String(e) });
  }
});

app.post("/save_model", async (_req, res) => {
  try {
    const r = await fetch(`${QUANTUM_URL}/save_model`, { method: "POST" });
    const text = await r.text();
    res.status(r.status).send(text);
  } catch (e) {
    res.status(502).json({ ok: false, error: String(e) });
  }
});

app.post("/load_model", async (_req, res) => {
  try {
    const r = await fetch(`${QUANTUM_URL}/load_model`, { method: "POST" });
    const text = await r.text();
    res.status(r.status).send(text);
  } catch (e) {
    res.status(502).json({ ok: false, error: String(e) });
  }
});

async function proxyJson(path, payload) {
  const resp = await fetch(`${QUANTUM_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await resp.text();
  if (!resp.ok) return { ok: false, status: resp.status, error: text };
  return { ok: true, data: JSON.parse(text) };
}

app.post("/train", async (req, res) => {
  const result = await proxyJson("/train", req.body);
  res.status(result.ok ? 200 : 502).json(result);
});

app.post("/predict", async (req, res) => {
  const resp = await fetch(`${QUANTUM_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body),
  });
  const text = await resp.text();
  res.status(resp.status).send(text);
});

app.post("/train_csv", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ ok: false, error: "Missing file" });

  try {
    const form = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype || "text/csv" });
    form.append("file", blob, req.file.originalname || "data.csv");

    for (const f of ["epochs", "lr", "seed", "label_col", "feature_cols", "test_ratio", "eval_every"]) {
      if (req.body?.[f] !== undefined) form.append(f, String(req.body[f]));
    }

    const r = await fetch(`${QUANTUM_URL}/train_csv`, { method: "POST", body: form });
    const text = await r.text();
    res.status(r.status).send(text);
  } catch (e) {
    res.status(502).json({ ok: false, error: String(e) });
  }
});

app.listen(PORT, () => {
  console.log(`Node API running: http://localhost:${PORT}`);
  console.log(`Proxying to: ${QUANTUM_URL}`);
});
