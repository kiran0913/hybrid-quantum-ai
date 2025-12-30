# Hybrid Quantum-AI Anomaly Detector (Web App)

A full-stack demo web application that trains and serves a **Hybrid Quantum-AI model** for **anomaly detection**.

**Frontend (HTML/CSS/JS)** → **Node.js API (Express)** → **Python FastAPI Quantum Service (PennyLane + PyTorch)**

✅ Features
- Train from CSV upload
- Auto-generated Predict inputs (based on model feature count)
- Training curve chart (loss + train/test accuracy)
- Threshold sweep chart (precision/recall/F1 vs threshold)
- Confusion matrix + evaluation metrics
- Save/Load model (persist training across restarts)
- Demo dataset download (Easy/Hard)

> Note: This is a prototype for learning + showcasing the pipeline (not production security software).

---

## Project Structure

```txt
hybrid-quantum-ai/
  api/
    server.js
    package.json
    package-lock.json
    .env.example
    .gitignore
  quantum-service/
    main.py
    requirements.txt
    saved_model/
    .gitignore
  web/
    index.html
    styles.css
    app.js
  README.md

Prerequisites

Windows
VS Code
Node.js (18+ recommended)
Python (recommended: 3.11–3.12)
VS Code extension: Live Server
If you use Python 3.14 and you hit package install issues, install Python 3.12 and use that for this project.

Setup & Run (3 terminals)
Terminal 1 — Python Quantum Service (FastAPI)

cd quantum-service
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload --port 8001

Test:

http://localhost:8001/health
→ {"ok":true}

Terminal 2 — Node API (Express proxy)

cd api
npm install
npm run dev

Test:

http://localhost:8000/health
→ { ok:true, quantum:"http://localhost:8001" }

Terminal 3 — Frontend (Live Server)

In VS Code:
Right-click web/index.html → Open with Live Server

Open in browser (example):
http://127.0.0.1:5500/web/index.html
If UI doesn’t update:
Hard refresh: Ctrl + Shift + R

How To Use
1) Train from CSV

In the UI → Train (Upload CSV):
Upload your CSV

Set:

Label Column: is_anomaly (or your label column)
Feature Columns: leave blank (auto = all except label)
Test Ratio: 0.2
Epochs: 150–250
LR: 0.01–0.03
Seed: 42
Eval Every: 2
Click Train from CSV
Wait for Status: READY

After training you’ll see:

Training chart
Confusion matrix
Precision/Recall/F1 metrics
Threshold sweep chart + best threshold

2) Predict

In the UI → Predict:
Set Threshold
Enter feature values (inputs are auto-generated)
Click Predict anomaly
Output meaning:
anomaly_score = probability (0..1)
is_anomaly = anomaly_score >= threshold

3) Save / Load model

Click Save model after training to persist it under quantum-service/saved_model/
After restarting Python, click Load model (or it auto-loads if saved files exist)

4) Download demo dataset

Use the Download HARD CSV / Download EASY CSV buttons, then upload that CSV in Train.

Config (.env)
Create api/.env from api/.env.example:
PORT=8000
QUANTUM_URL=http://localhost:8001

URLs Quick Reference
Frontend

Live Server URL, usually:

http://127.0.0.1:5500/web/index.html

Node API (8000)

http://localhost:8000/health
http://localhost:8000/status
http://localhost:8000/demo_dataset?mode=hard&rows=3000&anomaly_rate=0.05&seed=7

Python Quantum Service (8001)

http://localhost:8001/health
http://localhost:8001/status

