# F1 Telemetry & Team Radio AI Correlator 🏎️🎙️

An interactive command-line application that bridges the gap between vehicle kinematics and human emotion. This tool synchronizes live Formula 1 car telemetry (speed, throttle, braking) with official team radio audio streams, transcribing the communications locally using AI.

If a driver's radio feed is unavailable, the system automatically falls back to a custom **Kinematic Agitation Model**, calculating a physical stress score based on throttle volatility and brake locking.

## ✨ Features

* **Interactive CLI Engine:** Dynamically queries the OpenF1 API to allow users to select any Race, Year (2023+), and Driver.
* **Multi-Modal Data Synchronization:** Aligns FastF1 microsecond-level vehicle telemetry with OpenF1 audio timestamps.
* **Local AI Transcription:** Downloads `.mp3` radio streams into temporary memory and transcribes them locally using OpenAI's `Whisper` (Medium model).
* **Hallucination Prevention:** Uses context-priming prompts and deterministic parameters to prevent Whisper from hallucinating text during periods of pure V6 engine noise.
* **Fault-Tolerant Architecture:** Built-in safeguards handle missing API endpoints, zero-byte audio payloads, and corrupted FFmpeg audio frames without crashing.
* **Kinematic Agitation Fallback:** If API radio data is missing, the system calculates a driver's physical stress out of 10.0 by taking the derivative of their throttle and brake applications to detect erratic driving behavior.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Sources:** [FastF1](https://docs.fastf1.dev/) (Telemetry), [OpenF1 API](https://openf1.org/) (Live Timing & Radio)
* **Machine Learning:** [OpenAI Whisper](https://github.com/openai/whisper)
* **Data Engineering:** `pandas`, `requests`, `tempfile`
* **Audio Processing:** `FFmpeg`

