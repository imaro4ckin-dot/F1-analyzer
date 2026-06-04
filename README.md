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

## Architecture

```
F1 Analyzer/
├── app/
│   ├── dashboard.py      ← entry point: python app/dashboard.py
│   ├── layout.py         ← Dash layout, dark F1 theme
│   └── callbacks.py      ← all Dash callbacks + figure builders
├── core/
│   ├── data_loader.py    ← FastF1 + OpenF1 API wrappers
│   ├── predictor.py      ← continuous ML stress prediction
│   └── transcriber.py    ← Whisper audio transcription
├── ml/
│   ├── f1_stress_model.pkl         ← trained Random Forest model
│   ├── f1_all_seasons_training_data.csv
│   ├── collect_training_data.py    ← scrape + transcribe pipeline
│   └── train_model.py              ← train and save the model
├── cache/                ← FastF1 session cache (gitignored)
└── requirements.txt
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python app/dashboard.py
```

Open `http://localhost:8050` in your browser. Select a season, Grand Prix, and driver, then click **ANALYZE**.

> **Note:** First run downloads Whisper weights (~1.5 GB) and loads the FastF1 session into the cache. Subsequent runs for the same race are fast.

---

## AI Pipeline

### Continuous Stress Prediction
A Random Forest model (200 trees) trained on ~4 seasons of telemetry-radio pairs predicts driver stress from 8 kinematic features: average speed, max speed, speed variance, throttle volatility, throttle snap frequency, brake switch count, average RPM, max RPM.

Rather than predicting only at radio timestamps, the dashboard slides a 100-row window every 10 rows across the full telemetry dataset, producing a continuous stress curve at ~10Hz resolution. The result is a red filled area chart overlaid on the speed trace.

### Training the Model
The training labels are derived from VADER sentiment analysis on Whisper transcripts: negative/stressed speech maps to high stress scores, calm/positive speech maps to low scores. The model learns to reproduce these labels from the kinematic features alone — so it can predict stress even when there is no radio.

```bash
# Collect training data (requires network + time)
python ml/collect_training_data.py

# Train and save the model
python ml/train_model.py
```

### Radio Sync
Radio timestamps from OpenF1 are matched to FastF1 telemetry by absolute UTC timestamp. Track X/Y coordinates at each radio event are extracted from FastF1's position data stream and used to place markers on the map.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web dashboard | [Dash](https://dash.plotly.com/) + [Plotly](https://plotly.com/python/) |
| Telemetry | [FastF1](https://docs.fastf1.dev/) |
| Live timing & radio | [OpenF1 API](https://openf1.org/) |
| ML model | scikit-learn RandomForestRegressor |
| Transcription | [OpenAI Whisper](https://github.com/openai/whisper) (local, medium) |
| Data | pandas, numpy |

---

## Data Coverage

OpenF1 API radio data is available from the **2023 season onwards**. FastF1 telemetry goes back to 2018 but radio sync requires OpenF1, so the dashboard is limited to 2023+.
