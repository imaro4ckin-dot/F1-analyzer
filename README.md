# F1 Driver Stress Correlator

> *Telemetry tells you what the car did. Radio tells you how the driver felt about it.*

Most F1 data projects visualize car physics — speed traces, G-force envelopes, tyre degradation curves. This project does something different: it maps a driver's **psychological and structural stress** by cross-referencing telemetry anomalies with the official team radio audio clips that accompany them.

The core insight is temporal. A lock-up at Turn 1, a battery SoC drop, a sudden lift off throttle — these are data events. But layered on top of them is a human reaction: a sharp exhale, a frustrated "come on", a calm "box box, box box". This tool synchronizes those two streams and lets you explore both simultaneously.

---

## The Problem This Solves

Telemetry data is microsecond-precise but emotionally opaque. Team radio is emotionally rich but temporally imprecise. Neither source alone tells the full story of a difficult stint.

This project builds a **temporal index** that bridges the two: every radio message is anchored to the exact point in the car's telemetry timeline where it was transmitted, and the car's state at that moment — speed, throttle position, brake application — is surfaced alongside the transcript.

---

## Current Capabilities

### Multi-Modal Data Synchronization
Aligns FastF1 microsecond-level telemetry packets with OpenF1 audio timestamps. The alignment uses absolute timestamp matching rather than lap-relative offsets, making it robust across safety car periods and red flags where lap timing becomes unreliable.

### Local AI Transcription
Downloads `.mp3` radio streams into temporary memory and transcribes them locally using OpenAI Whisper (medium model). Transcription runs entirely on-device — no data leaves the machine, no API keys required for the audio pipeline.

**Hallucination prevention:** Whisper has a known failure mode where it echoes its own prompt back during silent audio (pure engine noise). The system uses a domain-specific context primer (`"Formula 1 team radio. [DRIVER] speaking to race engineer..."`) combined with `temperature=0.0` and `no_speech_threshold=0.6` to suppress this. Any transcript that contains its own prompt text is discarded and replaced with `[No audible human speech detected]`.

### Kinematic Agitation Fallback
When OpenF1 has no radio data for a driver/session combination, the system falls back to a **Kinematic Agitation Model**: it takes the derivative of throttle and brake inputs over a 500-sample window to detect erratic driving behavior and returns a normalized stress score out of 10.0.

This isn't a degraded experience — it's a parallel analytical lens. A driver with no radio messages but a kinematic agitation score of 8.4 is telling you something.

### Fault-Tolerant Architecture
- Zero-byte audio payloads are detected and skipped before Whisper sees them
- Corrupt FFmpeg frames are caught at the `RuntimeError` level without crashing the session
- API responses that return error dictionaries (e.g. `{"detail": "Not found"}`) are handled at every endpoint, not just the primary one
- Graceful driver code fallback when an invalid abbreviation is entered

---

## Roadmap: The Full Vision

The current CLI is the data engine. The planned dashboard is the interface that makes it legible.

### Anomaly Detection Layer
An algorithm that flags uncharacteristic telemetry events: sudden drops below expected speed at a given corner, G-force spikes that exceed the session's rolling baseline, battery SoC drops that suggest deployment strategy changes or failures. These anomalies become the anchors around which radio messages are clustered.

### Sentiment Mapping
Map transcripts to an NLP classifier to categorize each radio message: `Panic/Issue`, `Frustration/Traffic`, `Strategy Update`, `Driver Report`, `Acknowledgement`. These categories become color-coded markers plotted directly on the telemetry timeline — letting you see at a glance whether a difficult sector was mechanical, tactical, or psychological.

### Interactive Dashboard
A timeline-based visualization where:
- Clicking a telemetry anomaly jumps to the nearest radio clip and plays the audio
- Clicking a sentiment marker highlights the corresponding section of the telemetry trace
- A 2D track map overlay shows where each radio event occurred on-circuit

### Temporal Precision at Scale
The core architectural challenge is matching audio timestamps (second-level precision, UTC-anchored) to telemetry packets (10Hz, session-relative). The current implementation handles this. The dashboard will expose this indexing layer as a queryable API so the frontend can seek to arbitrary moments without re-processing.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Telemetry | [FastF1](https://docs.fastf1.dev/) |
| Live Timing & Radio | [OpenF1 API](https://openf1.org/) |
| Transcription | [OpenAI Whisper](https://github.com/openai/whisper) (local, medium model) |
| Audio Processing | FFmpeg |
| Data Engineering | `pandas`, `requests`, `tempfile` |

---

## Usage

```bash
python main.py
```

The script prompts interactively:

```
Enter the Year (e.g., 2023, 2024): 2024

Fetching available races for 2024 from OpenF1...

--- Available Races ---
1. Bahrain - Sakhir
2. Saudi Arabia - Jeddah
...

Select a race number: 5

--- Available Drivers ---
VER, PER, HAM, RUS, LEC, SAI, ...

Enter the 3-letter code of the driver to analyze: HAM
```

**Note:** OpenF1 API coverage begins in 2023. Earlier seasons will return a data limitation warning.

---

## Architecture Note

The temporal indexing problem is harder than it looks. FastF1 telemetry timestamps are session-relative and must be reconstructed to UTC before they can be matched against OpenF1's absolute timestamps. DST handling, session timezone offsets, and the gap between qualifying and race sessions all introduce alignment errors if the reconstruction is naive. The current implementation anchors on session start time from FastF1's metadata and applies the offset uniformly across the telemetry frame before alignment.
