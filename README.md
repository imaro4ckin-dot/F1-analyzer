# F1 Driver Stress Correlator

> *Telemetry tells you what the car did. Radio tells you how the driver felt about it.*

An interactive web dashboard that maps a driver's **psychological and structural stress** by correlating continuous AI predictions over raw telemetry with team radio transmissions — anchored to the exact track position where they were broadcast.

---

## Dashboard

```
┌── F1 STRESS ANALYZER ─────────────────────────────────────────────────────┐
│  [2024] [Monaco · Monte-Carlo] [LEC]  [ANALYZE]   18 radio events · 42k pts│
├────────────────────────────────────┬───────────────────────────────────────┤
│  TELEMETRY & AI STRESS             │  TRACK MAP · RADIO EVENTS             │
│                                    │                                        │
│  Speed ────────────────────────    │       ╭────────────────╮               │
│  Throttle ·····················    │       │   ● ●          │               │
│  AI Stress ████████████████████    │       │         ●      │               │
│  ▼ Brake events                    │       │  ╭─────╯  ●    │               │
│  ◆ Radio marker (clickable)        │       ╰──╯             │               │
│                                    │    ● = radio event (colour = stress)   │
├────────────────────────────────────┴───────────────────────────────────────┤
│  RADIO TRANSMISSION  15:42:11                                               │
│  AI STRESS: ████████░░  7.4/10                                             │
│  "Box box box, come on, these tyres are gone"                              │
│  [▶ Audio player]                                                           │
└────────────────────────────────────────────────────────────────────────────┘
```

Click any **red circle on the track map** or **diamond marker on the telemetry chart** to reveal the radio panel: transcript, AI stress score, and inline audio playback.

---

## What Makes This Different

Most F1 data projects visualize car physics. This one correlates them with human state. Every radio message is placed at the exact X/Y coordinate on the track where it was transmitted. The AI stress curve runs continuously across the entire race — not just at radio timestamps — so you can see the build-up before a driver speaks.

---

## Requirements

- Python 3.10 or newer
- ~2 GB free disk space (Whisper model weights, downloaded on first run)
- ~4 GB free disk space (FastF1 session cache, grows as you load races)
- Internet connection (FastF1 + OpenF1 API calls)

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/f1-analyzer.git
cd f1-analyzer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** If `openai-whisper` fails to install, make sure you have [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH.

> **macOS users:** If you get SSL certificate errors on first run, execute:
> ```bash
> /Applications/Python\ 3.x/Install\ Certificates.command
> ```
> Or set `DISABLE_SSL_VERIFY=true` in your `.env` file as a workaround (see step 4).

### 4. Configure environment (optional)

Copy the example env file and edit if needed:

```bash
cp .env.example .env
```

The defaults work out of the box. The only setting you may need to change on macOS is `DISABLE_SSL_VERIFY=true` if you hit SSL errors.

### 5. Run the dashboard

```bash
python app/dashboard.py
```

Open `http://localhost:8050` in your browser.

Select a **season**, **Grand Prix**, and **driver**, then click **ANALYZE**.

> **First run:** Whisper downloads ~1.5 GB of model weights and FastF1 fetches the session into the local cache. This can take a few minutes. Every subsequent run for the same race is fast.

---

## Run with Docker

If you have [Docker](https://www.docker.com/) installed you can skip the Python setup entirely:

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:8050`. The FastF1 cache is stored in a named Docker volume and persists between restarts.

---

## Architecture

```
F1 Analyzer/
├── app/
│   ├── dashboard.py      ← entry point
│   ├── layout.py         ← Dash layout, dark F1 theme
│   ├── callbacks.py      ← all Dash callbacks + figure builders
│   └── race_page.py      ← animated race overview page
├── core/
│   ├── data_loader.py    ← FastF1 + OpenF1 API wrappers
│   ├── predictor.py      ← continuous ML stress prediction
│   └── transcriber.py    ← Whisper audio transcription
├── ml/
│   ├── f1_stress_model.pkl              ← trained Random Forest (included)
│   ├── f1_all_seasons_training_data.csv ← training dataset (included)
│   ├── collect_training_data.py         ← scrape + transcribe pipeline
│   └── train_model.py                   ← train and save the model
├── assets/
│   └── style.css         ← Plotly + Dash CSS overrides
├── cache/                ← FastF1 session cache (gitignored, auto-created)
├── .env.example          ← environment variable template
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## AI Pipeline

### Continuous Stress Prediction

A Random Forest model (200 trees) trained on ~4 seasons of telemetry-radio pairs predicts driver stress from 8 kinematic features: average speed, max speed, speed variance, throttle volatility, throttle snap frequency, brake switch count, average RPM, max RPM.

Rather than predicting only at radio timestamps, the dashboard slides a 100-row window every 10 rows across the full telemetry dataset, producing a continuous stress curve at ~10 Hz resolution. The result is a red filled area chart overlaid on the speed trace.

### Training the Model (optional)

The model is included pre-trained (`ml/f1_stress_model.pkl`). If you want to retrain it from scratch:

```bash
# Step 1 — collect training data (requires network + significant time)
python ml/collect_training_data.py

# Step 2 — train and save the model
python ml/train_model.py
```

Training labels come from VADER sentiment analysis on Whisper transcripts: negative/stressed speech maps to high stress scores, calm/positive speech maps to low scores. The model learns to reproduce these labels from kinematic features alone — so it can predict stress even when there is no radio.

### Radio Sync

Radio timestamps from OpenF1 are matched to FastF1 telemetry by absolute UTC timestamp. Track X/Y coordinates at each radio event are extracted from FastF1's position data stream and used to place markers on the map.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web dashboard | [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/python/) |
| UI framework | [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) |
| Telemetry | [FastF1](https://docs.fastf1.dev/) |
| Live timing & radio | [OpenF1 API](https://openf1.org/) |
| ML model | scikit-learn RandomForestRegressor |
| Transcription | [OpenAI Whisper](https://github.com/openai/whisper) (local, medium) |
| Data | pandas, numpy |
| Production server | Gunicorn |

---

## Data Coverage

OpenF1 API radio data is available from the **2023 season onwards**. FastF1 telemetry goes back to 2018 but radio sync requires OpenF1, so the dashboard is limited to 2023+.

---

## Troubleshooting

**`ModuleNotFoundError` on startup**
Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**SSL errors on macOS**
Run `/Applications/Python 3.x/Install Certificates.command` or set `DISABLE_SSL_VERIFY=true` in your `.env`.

**App is slow on first race load**
FastF1 downloads and caches the session on first access. Subsequent loads for the same race are instant.

**Whisper download takes forever**
The medium model is ~1.5 GB. It downloads once and is cached by Whisper automatically.

**Port 8050 already in use**
Set a different port in your `.env`: `PORT=8060`
