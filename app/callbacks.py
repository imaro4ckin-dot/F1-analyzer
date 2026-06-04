import sys
import os
import json
import threading

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, dcc
from dash.exceptions import PreventUpdate

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_loader import fetch_races, load_session, get_driver_map, get_telemetry, get_track_coords, fetch_radio
from core.predictor import load_model, predict_continuous, predict_at_timestamp
from core.transcriber import transcribe_audio_url
from app.layout import BG, SURFACE, CARD, BORDER, RED, RED_DIM, WHITE, GREY, GREEN, ORANGE, LABEL_STYLE

# ── Load ML model once at import time ───────────────────────────────────────
_MODEL = None
_MODEL_LOCK = threading.Lock()


def _get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = load_model()
    return _MODEL


# ── Whisper model (lazy, background) ────────────────────────────────────────
_WHISPER = None
_WHISPER_LOCK = threading.Lock()


def _get_whisper():
    global _WHISPER
    with _WHISPER_LOCK:
        if _WHISPER is None:
            import whisper
            _WHISPER = whisper.load_model("medium")
    return _WHISPER


def register_callbacks(app):
    # ── 1. Populate race dropdown when year changes ──────────────────────────
    @app.callback(
        Output("dd-race", "options"),
        Output("dd-race", "value"),
        Input("dd-year", "value"),
    )
    def update_races(year):
        if not year:
            raise PreventUpdate
        races = fetch_races(int(year))
        options = [{"label": f"{r['country_name']}  ·  {r['location']}", "value": r["location"]}
                   for r in races]
        value = options[0]["value"] if options else None
        return options, value

    # ── 2. Run full analysis pipeline on ANALYZE click ───────────────────────
    @app.callback(
        Output("chart-telemetry", "figure"),
        Output("chart-track", "figure"),
        Output("store-radio", "data"),
        Output("store-session-meta", "data"),
        Output("status-msg", "children"),
        Input("btn-analyze", "n_clicks"),
        State("dd-year", "value"),
        State("dd-race", "value"),
        State("input-driver", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, year, race_location, driver_code):
        if not all([year, race_location, driver_code]):
            raise PreventUpdate

        driver_code = driver_code.strip().upper()

        # ── Load session ────────────────────────────────────────────────────
        try:
            session = load_session(int(year), race_location)
        except Exception as e:
            return _err_fig(), _err_fig(), None, None, f"Session load failed: {e}"

        driver_map = get_driver_map(session)
        if driver_code not in driver_map:
            return _err_fig(), _err_fig(), None, None, f"Driver '{driver_code}' not found in this session."

        driver_num = driver_map[driver_code]

        # ── Telemetry ───────────────────────────────────────────────────────
        try:
            tel = get_telemetry(session, driver_code)
        except Exception as e:
            return _err_fig(), _err_fig(), None, None, f"Telemetry error: {e}"

        # ── Continuous AI stress curve ──────────────────────────────────────
        model = _get_model()
        stress = predict_continuous(tel, model)

        # ── Track position data ─────────────────────────────────────────────
        try:
            pos = get_track_coords(session, driver_code)
        except Exception:
            pos = pd.DataFrame(columns=["Date", "X", "Y"])

        # ── Radio data ──────────────────────────────────────────────────────
        races = fetch_races(int(year))
        session_key = next((r["session_key"] for r in races if r["location"] == race_location), None)
        radio_messages = fetch_radio(session_key, driver_num) if session_key else []

        # Build enriched radio records (stress + track position + transcript)
        radio_records = []
        whisper_model = _get_whisper()

        for msg in radio_messages:
            radio_time = pd.to_datetime(msg["date"]).tz_localize(None)
            audio_url = msg.get("recording_url", "")

            # Stress at this moment
            stress_val = predict_at_timestamp(tel, radio_time, model)

            # Track position at this moment
            rx, ry = None, None
            if not pos.empty:
                closest_pos_idx = (pos["Date"] - radio_time).abs().argsort().iloc[0]
                row = pos.iloc[closest_pos_idx]
                rx, ry = float(row["X"]), float(row["Y"])

            # Transcribe
            transcript = ""
            if audio_url:
                transcript = transcribe_audio_url(audio_url, driver_code, whisper_model)

            radio_records.append({
                "time": str(radio_time),
                "stress": stress_val,
                "audio_url": audio_url,
                "transcript": transcript or "[engine static]",
                "x": rx,
                "y": ry,
            })

        # ── Build telemetry figure ──────────────────────────────────────────
        tel_fig = _build_telemetry_figure(tel, stress, radio_records, driver_code)

        # ── Build track map figure ──────────────────────────────────────────
        track_fig = _build_track_figure(pos, radio_records)

        meta = {"year": year, "location": race_location, "driver": driver_code}
        status = f"{len(radio_records)} radio events  ·  {len(tel)} telemetry points"
        return tel_fig, track_fig, radio_records, meta, status

    # ── 3. Show radio panel on marker click ─────────────────────────────────
    @app.callback(
        Output("radio-panel", "children"),
        Input("chart-track", "clickData"),
        Input("chart-telemetry", "clickData"),
        State("store-radio", "data"),
        prevent_initial_call=True,
    )
    def show_radio_panel(track_click, tel_click, radio_records):
        if not radio_records:
            raise PreventUpdate

        # Determine which chart fired and extract the custom data index
        click_data = track_click or tel_click
        if not click_data:
            raise PreventUpdate

        points = click_data.get("points", [])
        if not points:
            raise PreventUpdate

        point = points[0]
        custom = point.get("customdata")

        # customdata is the index into radio_records list
        if custom is None:
            raise PreventUpdate

        idx = int(custom)
        if idx < 0 or idx >= len(radio_records):
            raise PreventUpdate

        rec = radio_records[idx]
        return _build_radio_panel(rec)


# ── Figure builders ──────────────────────────────────────────────────────────

def _base_fig():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": "JetBrains Mono, Courier New, monospace", "size": 11},
        margin={"l": 50, "r": 20, "t": 20, "b": 40},
        legend={"bgcolor": "rgba(0,0,0,0)", "bordercolor": BORDER, "font": {"size": 10}},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont={"color": GREY})
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont={"color": GREY})
    return fig


def _err_fig():
    from app.layout import _empty_figure
    return _empty_figure("Error loading data")


def _build_telemetry_figure(tel, stress, radio_records, driver_code):
    fig = _base_fig()
    fig.update_layout(hovermode="x unified")

    x = tel["Date"]

    # Speed (white, primary y-axis)
    fig.add_trace(go.Scatter(
        x=x, y=tel["Speed"],
        name="Speed (km/h)",
        line={"color": WHITE, "width": 1.5},
        yaxis="y",
        hovertemplate="%{y:.0f} km/h",
    ))

    # Throttle (green, secondary y-axis)
    fig.add_trace(go.Scatter(
        x=x, y=tel["Throttle"],
        name="Throttle %",
        line={"color": GREEN, "width": 1, "dash": "dot"},
        yaxis="y2",
        opacity=0.7,
        hovertemplate="%{y:.0f}%",
    ))

    # AI Stress (red filled area, tertiary y-axis 0-10)
    fig.add_trace(go.Scatter(
        x=x, y=stress,
        name="AI Stress",
        fill="tozeroy",
        fillcolor="rgba(225,6,0,0.15)",
        line={"color": RED, "width": 2},
        yaxis="y3",
        hovertemplate="Stress: %{y:.1f}/10",
    ))

    # Brake events (orange dots on speed line)
    brake_mask = tel["Brake"].astype(bool)
    fig.add_trace(go.Scatter(
        x=x[brake_mask], y=tel["Speed"][brake_mask],
        name="Braking",
        mode="markers",
        marker={"color": ORANGE, "size": 3, "symbol": "circle"},
        yaxis="y",
        hovertemplate="Braking @ %{y:.0f} km/h",
    ))

    # Radio vertical lines + markers
    for i, rec in enumerate(radio_records):
        rt = pd.to_datetime(rec["time"])
        transcript_short = rec["transcript"][:40] + ("…" if len(rec["transcript"]) > 40 else "")
        fig.add_vline(
            x=rt,
            line={"color": RED, "width": 1, "dash": "dash"},
            opacity=0.5,
        )
        # Invisible scatter point to make vlines clickable
        fig.add_trace(go.Scatter(
            x=[rt],
            y=[rec["stress"]],
            mode="markers",
            marker={"color": RED, "size": 10, "symbol": "diamond", "line": {"color": WHITE, "width": 1}},
            name=f"Radio #{i+1}",
            yaxis="y3",
            customdata=[i],
            hovertemplate=f"<b>Radio #{i+1}</b><br>{transcript_short}<br>Stress: {rec['stress']:.1f}/10<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        yaxis={"title": {"text": "Speed (km/h)", "font": {"color": WHITE, "size": 10}}, "side": "left"},
        yaxis2={"title": {"text": "Throttle %", "font": {"color": GREEN, "size": 10}},
                "overlaying": "y", "side": "right", "range": [0, 110], "showgrid": False},
        yaxis3={"title": {"text": "Stress /10", "font": {"color": RED, "size": 10}},
                "overlaying": "y", "side": "right", "range": [0, 10], "showgrid": False,
                "anchor": "free", "position": 0.97},
        title={"text": f"{driver_code}  ·  Telemetry + AI Stress", "font": {"size": 12, "color": GREY}, "x": 0},
    )
    return fig


def _build_track_figure(pos, radio_records):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": "JetBrains Mono, Courier New, monospace", "size": 10},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False,
               "scaleanchor": "y", "scaleratio": 1},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        hovermode="closest",
        showlegend=False,
    )

    # Track outline
    if not pos.empty:
        fig.add_trace(go.Scatter(
            x=pos["X"], y=pos["Y"],
            mode="lines",
            line={"color": "#3A3A3A", "width": 8},
            hoverinfo="skip",
            name="Track",
        ))
        # Thin white centre line
        fig.add_trace(go.Scatter(
            x=pos["X"], y=pos["Y"],
            mode="lines",
            line={"color": "#555555", "width": 2},
            hoverinfo="skip",
            name="Centre line",
        ))

    # Radio event markers
    valid = [(i, r) for i, r in enumerate(radio_records) if r.get("x") is not None and r.get("y") is not None]
    if valid:
        xs = [r["x"] for _, r in valid]
        ys = [r["y"] for _, r in valid]
        indices = [i for i, _ in valid]
        transcripts = [r["transcript"][:50] + ("…" if len(r["transcript"]) > 50 else "") for _, r in valid]
        stresses = [r["stress"] for _, r in valid]
        times = [pd.to_datetime(r["time"]).strftime("%H:%M:%S") for _, r in valid]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker={
                "color": [_stress_colour(s) for s in stresses],
                "size": 14,
                "symbol": "circle",
                "line": {"color": WHITE, "width": 1.5},
            },
            customdata=indices,
            hovertemplate=(
                "<b>Radio Event</b><br>"
                "Time: %{customdata}<br>"
                "<extra></extra>"
            ),
            text=[f"{t}<br>{tr}" for t, tr in zip(times, transcripts)],
            hovertext=[f"{t}  |  Stress {s:.1f}/10<br>{tr}" for t, s, tr in zip(times, stresses, transcripts)],
            hoverinfo="text",
            name="Radio",
        ))

        # Pulse ring around markers (outer glow effect)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker={
                "color": "rgba(0,0,0,0)",
                "size": 22,
                "symbol": "circle-open",
                "line": {"color": RED, "width": 1},
                "opacity": 0.4,
            },
            customdata=indices,
            hoverinfo="skip",
            showlegend=False,
        ))

    return fig


def _stress_colour(stress: float) -> str:
    """Map a stress score 1–10 to a colour between green and red."""
    t = (stress - 1.0) / 9.0  # normalise to [0, 1]
    t = max(0.0, min(1.0, t))
    r = int(39 + (225 - 39) * t)
    g = int(255 + (6 - 255) * t)
    b = int(20 + (0 - 20) * t)
    return f"rgb({r},{g},{b})"


def _build_radio_panel(rec):
    stress = rec["stress"]
    stress_colour = _stress_colour(stress)
    time_str = pd.to_datetime(rec["time"]).strftime("%H:%M:%S")
    transcript = rec["transcript"]
    audio_url = rec.get("audio_url", "")

    # Stress bar (percentage of 10)
    bar_pct = int((stress / 10) * 100)

    return html.Div(
        className="radio-card",
        style={
            "backgroundColor": CARD,
            "border": f"1px solid {RED}",
            "borderLeft": f"4px solid {RED}",
            "borderRadius": "6px",
            "padding": "20px 24px",
            "display": "flex",
            "gap": "32px",
            "alignItems": "center",
            "flexWrap": "wrap",
        },
        children=[
            # Left: metadata
            html.Div(style={"minWidth": "180px"}, children=[
                html.Div("RADIO TRANSMISSION", style={**LABEL_STYLE, "marginBottom": "8px"}),
                html.Div(time_str, style={"color": WHITE, "fontSize": "24px", "fontWeight": "700",
                                         "letterSpacing": "0.1em"}),
                html.Div(style={"marginTop": "12px"}, children=[
                    html.Div("AI STRESS LEVEL", style={**LABEL_STYLE, "marginBottom": "6px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                        html.Div(style={
                            "height": "6px", "width": "120px", "backgroundColor": BORDER, "borderRadius": "3px",
                        }, children=[
                            html.Div(style={
                                "height": "6px", "width": f"{bar_pct}%",
                                "backgroundColor": stress_colour, "borderRadius": "3px",
                                "transition": "width 0.5s ease",
                            })
                        ]),
                        html.Span(f"{stress:.1f}/10", style={"color": stress_colour, "fontSize": "13px",
                                                              "fontWeight": "700"}),
                    ]),
                ]),
            ]),

            # Centre: transcript
            html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                html.Div("TRANSCRIPT", style={**LABEL_STYLE, "marginBottom": "8px"}),
                html.Div(
                    f'"{transcript}"',
                    style={
                        "color": WHITE, "fontSize": "14px", "lineHeight": "1.6",
                        "fontStyle": "italic" if transcript != "[engine static]" else "normal",
                        "color": GREY if transcript == "[engine static]" else WHITE,
                    },
                ),
            ]),

            # Right: audio player
            html.Div(style={"minWidth": "200px"}, children=[
                html.Div("AUDIO", style={**LABEL_STYLE, "marginBottom": "8px"}),
                html.Audio(
                    src=audio_url,
                    controls=True,
                    style={
                        "width": "100%",
                        "filter": "invert(1) hue-rotate(150deg) brightness(0.9)",
                    },
                ) if audio_url else html.Div("No audio available", style={"color": GREY, "fontSize": "11px"}),
            ]),
        ],
    )
