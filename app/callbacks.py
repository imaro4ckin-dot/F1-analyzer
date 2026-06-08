import sys
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, dcc
from dash.exceptions import PreventUpdate

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_loader import (
    fetch_races, load_session, get_driver_map, get_telemetry, get_track_coords,
    fetch_radio, get_stint_data, fetch_race_control, get_all_driver_codes,
    get_lap_times_data, get_pit_stops,
)
from core.predictor import load_model, predict_continuous, predict_at_timestamp, predict_anomaly_zscore
from core.transcriber import transcribe_audio_url
from app.layout import (
    BG, SURFACE, CARD, BORDER, RED, RED_DIM, WHITE, GREY, GREEN, ORANGE,
    YELLOW, TYRE_SOFT, TYRE_MEDIUM, TYRE_HARD, LABEL_STYLE, FONT_MONO,
)

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
        # Build options from FastF1 event schedule (local/cached, always works).
        # Fall back to OpenF1 API if FastF1 schedule is unavailable.
        options = []
        try:
            import fastf1
            schedule = fastf1.get_event_schedule(int(year), include_testing=False)
            for _, ev in schedule.iterrows():
                loc = ev.get("Location") or ev.get("EventName", "")
                country = ev.get("Country", "")
                if not loc:
                    continue
                label = f"{country}  ·  {loc}" if country else loc
                options.append({"label": label, "value": loc})
        except Exception:
            pass

        if not options:
            # OpenF1 fallback
            races = fetch_races(int(year))
            options = [
                {"label": f"{r['country_name']}  ·  {r['location']}", "value": r["location"]}
                for r in races
            ]

        value = options[0]["value"] if options else None
        return options, value

    # ── 2. Run full analysis pipeline on ANALYZE click ───────────────────────
    @app.callback(
        Output("chart-telemetry", "figure"),
        Output("chart-track", "figure"),
        Output("store-radio", "data"),
        Output("store-session-meta", "data"),
        Output("status-msg", "children"),
        Output("store-lap-stress", "data"),
        Output("store-incidents", "data"),
        Output("store-lap-times", "data"),
        Input("btn-analyze", "n_clicks"),
        State("dd-year", "value"),
        State("dd-race", "value"),
        State("input-driver", "value"),
        State("input-driver2", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, year, race_location, driver_code, driver_code2):
        if not all([year, race_location, driver_code]):
            raise PreventUpdate

        driver_code = driver_code.strip().upper()
        driver_code2 = driver_code2.strip().upper() if driver_code2 and driver_code2.strip() else None

        # ── Load session ────────────────────────────────────────────────────
        try:
            session = load_session(int(year), race_location)
        except Exception as e:
            return _err_fig(), _err_fig(), None, None, f"Session load failed: {e}", None, None, None

        driver_map = get_driver_map(session)
        if driver_code not in driver_map:
            return _err_fig(), _err_fig(), None, None, f"Driver '{driver_code}' not found.", None, None, None

        driver_num = driver_map[driver_code]

        # ── Parallel: HTTP calls + local data extraction ─────────────────────
        # Get session_key from already-loaded FastF1 session (no OpenF1 call needed)
        try:
            session_key = int(session.session_info["Key"])
        except Exception:
            session_key = None

        with ThreadPoolExecutor(max_workers=6) as pool:
            f_tel      = pool.submit(get_telemetry, session, driver_code)
            f_pos      = pool.submit(get_track_coords, session, driver_code)
            f_stint    = pool.submit(get_stint_data, session, driver_code)

            # Fire HTTP calls now that we have session_key
            f_radio    = pool.submit(fetch_radio, session_key, driver_num) if session_key else None
            f_rc       = pool.submit(fetch_race_control, session_key) if session_key else None

            try:
                tel = f_tel.result()
            except Exception as e:
                return _err_fig(), _err_fig(), None, None, f"Telemetry error: {e}", None, None, None

            try:
                pos = f_pos.result()
            except Exception:
                pos = pd.DataFrame(columns=["Date", "X", "Y"])

            stint_data     = f_stint.result()
            radio_messages = f_radio.result() if f_radio else []
            incidents      = f_rc.result()   if f_rc   else []

        # ── Model & stress driver 1 ─────────────────────────────────────────
        model = _get_model()
        has_radio = len(radio_messages) > 0
        stress = predict_continuous(tel, model, use_anomaly=not has_radio)
        stress_mode = "RF Model" if has_radio else "Anomaly Z-Score"

        # ── Radio records driver 1 ──────────────────────────────────────────
        radio_records = []
        whisper_model = _get_whisper()
        for msg in radio_messages:
            radio_time = pd.to_datetime(msg["date"]).tz_localize(None)
            audio_url = msg.get("recording_url", "")
            stress_val = predict_at_timestamp(tel, radio_time, model)
            rx, ry = None, None
            if not pos.empty:
                closest_pos_idx = (pos["Date"] - radio_time).abs().idxmin()
                row = pos.loc[closest_pos_idx]
                rx, ry = float(row["X"]), float(row["Y"])
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
                "driver": "D1",
                "code": driver_code,
            })

        # ── Driver 2 (optional) ─────────────────────────────────────────────
        tel2 = stress2 = pos2 = None
        radio_records2 = []
        if driver_code2 and driver_code2 in driver_map:
            try:
                driver_num2 = driver_map[driver_code2]
                tel2 = get_telemetry(session, driver_code2)
                try:
                    pos2 = get_track_coords(session, driver_code2)
                except Exception:
                    pos2 = pd.DataFrame(columns=["Date", "X", "Y"])
                radio_msgs2 = fetch_radio(session_key, driver_num2) if session_key else []
                has_radio2 = len(radio_msgs2) > 0
                stress2 = predict_continuous(tel2, model, use_anomaly=not has_radio2)
                for msg in radio_msgs2:
                    radio_time2 = pd.to_datetime(msg["date"]).tz_localize(None)
                    audio_url2 = msg.get("recording_url", "")
                    stress_val2 = predict_at_timestamp(tel2, radio_time2, model)
                    rx2, ry2 = None, None
                    if not pos2.empty:
                        idx2 = (pos2["Date"] - radio_time2).abs().idxmin()
                        row2 = pos2.loc[idx2]
                        rx2, ry2 = float(row2["X"]), float(row2["Y"])
                    transcript2 = ""
                    if audio_url2:
                        transcript2 = transcribe_audio_url(audio_url2, driver_code2, whisper_model)
                    radio_records2.append({
                        "time": str(radio_time2),
                        "stress": stress_val2,
                        "audio_url": audio_url2,
                        "transcript": transcript2 or "[engine static]",
                        "x": rx2,
                        "y": ry2,
                        "driver": "D2",
                        "code": driver_code2,
                    })
            except Exception:
                tel2 = stress2 = pos2 = None
                radio_records2 = []

        # ── Merge radio records ─────────────────────────────────────────────
        all_radio = radio_records + radio_records2

        # ── Lap stress breakdown ────────────────────────────────────────────
        lap_stress_data = _compute_lap_stress(tel, stress, stint_data)

        # ── Fastest lap / pit stops / lap times ─────────────────────────────
        fastest_lap_band = _get_fastest_lap_band(session, driver_code)
        pit_stops        = get_pit_stops(session, driver_code)
        lap_times_data   = get_lap_times_data(session, driver_code)

        # Format fastest lap as M:SS.sss for stats bar
        fl_str = "—"
        if fastest_lap_band:
            lt_s = fastest_lap_band["lap_time_s"]
            fl_str = f"{int(lt_s // 60)}:{lt_s % 60:06.3f}"

        stress_vals = [r["avg_stress"] for r in lap_stress_data]
        lap_store = {
            "driver_code": driver_code,
            "laps": lap_stress_data,
            "fastest_lap_num": fastest_lap_band.get("lap_num") if fastest_lap_band else None,
            "stats": {
                "peak_stress":  round(max(stress_vals, default=0.0), 1),
                "avg_stress":   round(sum(stress_vals) / len(stress_vals), 1) if stress_vals else 0.0,
                "fastest_lap":  fl_str,
                "pit_stops":    len(pit_stops),
                "radio_count":  len(all_radio),
            },
        }

        # ── Build figures ───────────────────────────────────────────────────
        tel_fig = _build_telemetry_figure(
            tel, stress, radio_records, driver_code, stress_mode,
            tel2=tel2, stress2=stress2, radio_records2=radio_records2, driver_code2=driver_code2,
            stint_data=stint_data, incidents=incidents,
            pit_stops=pit_stops, fastest_lap_band=fastest_lap_band,
        )
        track_fig = _build_track_figure(
            pos, radio_records,
            pos2=pos2, radio_records2=radio_records2, driver_code2=driver_code2,
            incidents=incidents,
        )

        meta = {"year": year, "location": race_location, "driver": driver_code}
        d2_label = f"  ·  vs {driver_code2}" if driver_code2 else ""
        status = (
            f"{len(all_radio)} radio events  ·  {len(tel)} telemetry points  ·  "
            f"AI: {stress_mode}{d2_label}"
        )
        return tel_fig, track_fig, all_radio, meta, status, lap_store, incidents, lap_times_data

    # ── 3. Compute leaderboard separately (triggered after session meta ready) ─
    @app.callback(
        Output("store-leaderboard", "data"),
        Input("store-session-meta", "data"),
        State("dd-year", "value"),
        State("dd-race", "value"),
        prevent_initial_call=True,
    )
    def compute_leaderboard(meta, year, race_location):
        if not meta or not year or not race_location:
            raise PreventUpdate
        try:
            session = load_session(int(year), race_location)
            model = _get_model()
            return _compute_leaderboard(session, model)
        except Exception:
            return []

    # ── 4. Lap bar chart ──────────────────────────────────────────────────────
    @app.callback(
        Output("chart-lap-stress", "figure"),
        Input("store-lap-stress", "data"),
        prevent_initial_call=True,
    )
    def update_lap_chart(lap_stress_data):
        if not lap_stress_data or not lap_stress_data.get("laps"):
            raise PreventUpdate
        laps = lap_stress_data["laps"]
        if not laps:
            raise PreventUpdate

        peak_lap_entry = max(laps, key=lambda r: r["avg_stress"])
        peak_lap = peak_lap_entry["lap"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[r["lap"] for r in laps],
            y=[r["avg_stress"] for r in laps],
            marker_color=[_stress_colour(r["avg_stress"]) for r in laps],
            customdata=[r["lap"] for r in laps],
            hovertemplate="Lap %{x}  ·  Stress %{y:.2f}/10<extra></extra>",
            name="Avg Stress",
        ))

        # Highlight peak lap with a red border shape
        fig.add_shape(
            type="rect",
            x0=peak_lap - 0.5, x1=peak_lap + 0.5,
            y0=0, y1=10,
            xref="x", yref="y",
            line={"color": RED, "width": 2},
            fillcolor="rgba(0,0,0,0)",
        )

        fig.add_annotation(
            x=peak_lap, y=peak_lap_entry["avg_stress"],
            text="PEAK",
            showarrow=False,
            yshift=14,
            font={"color": RED, "size": 9, "family": "JetBrains Mono, Courier New, monospace"},
        )

        # Fastest lap badge — green border + label (only if different from peak stress lap)
        fastest_lap_num = lap_stress_data.get("fastest_lap_num")
        if fastest_lap_num and fastest_lap_num != peak_lap:
            fl_entry = next((r for r in laps if r["lap"] == fastest_lap_num), None)
            if fl_entry:
                fig.add_shape(
                    type="rect",
                    x0=fastest_lap_num - 0.5, x1=fastest_lap_num + 0.5,
                    y0=0, y1=10,
                    xref="x", yref="y",
                    line={"color": GREEN, "width": 2},
                    fillcolor="rgba(57,255,20,0.08)",
                )
                fig.add_annotation(
                    x=fastest_lap_num, y=fl_entry["avg_stress"],
                    text="FASTEST",
                    showarrow=False,
                    yshift=14,
                    font={"color": GREEN, "size": 9,
                          "family": "JetBrains Mono, Courier New, monospace"},
                )

        fig.update_layout(
            paper_bgcolor=CARD,
            plot_bgcolor=CARD,
            font={"color": WHITE, "family": "JetBrains Mono, Courier New, monospace", "size": 10},
            margin={"l": 50, "r": 10, "t": 10, "b": 30},
            xaxis={
                "title": {"text": "Lap", "font": {"color": GREY, "size": 11}},
                "gridcolor": "#333333", "griddash": "dot", "zerolinecolor": BORDER,
                "tickfont": {"color": GREY, "size": 10},
                "dtick": 5,
            },
            yaxis={
                "title": {"text": "Stress Score", "font": {"color": RED, "size": 11}},
                "gridcolor": "#333333", "griddash": "dot", "zerolinecolor": BORDER,
                "tickfont": {"color": GREY, "size": 10},
                "range": [0, 10.5],
            },
            showlegend=False,
            hovermode="x",
            bargap=0.2,
        )
        return fig

    # ── 5. Clientside zoom: lap bar click → telemetry zoom ───────────────────
    app.clientside_callback(
        """
        function(clickData, lapData) {
            if (!clickData || !lapData || !lapData.laps) {
                return window.dash_clientside.no_update;
            }
            var lapNum = clickData.points[0].customdata;
            var laps = lapData.laps;
            var entry = null;
            for (var i = 0; i < laps.length; i++) {
                if (laps[i].lap === lapNum) { entry = laps[i]; break; }
            }
            if (!entry) return window.dash_clientside.no_update;
            var graphDiv = document.getElementById('chart-telemetry');
            if (graphDiv && window.Plotly) {
                Plotly.relayout(graphDiv, {
                    'xaxis.range[0]': entry.start_time,
                    'xaxis.range[1]': entry.end_time,
                    'xaxis2.range[0]': entry.start_time,
                    'xaxis2.range[1]': entry.end_time,
                    'xaxis3.range[0]': entry.start_time,
                    'xaxis3.range[1]': entry.end_time
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("chart-lap-stress", "id"),
        Input("chart-lap-stress", "clickData"),
        State("store-lap-stress", "data"),
    )

    # ── 6. Leaderboard display ────────────────────────────────────────────────
    @app.callback(
        Output("leaderboard-panel", "children"),
        Input("store-leaderboard", "data"),
        prevent_initial_call=True,
    )
    def update_leaderboard(leaderboard_data):
        if not leaderboard_data:
            raise PreventUpdate
        return _build_leaderboard_panel(leaderboard_data)

    # ── 7. Lap time evolution chart ───────────────────────────────────────────
    @app.callback(
        Output("chart-lap-times", "figure"),
        Input("store-lap-times", "data"),
        prevent_initial_call=True,
    )
    def update_lap_time_chart(lap_times_data):
        if not lap_times_data:
            raise PreventUpdate
        return _build_lap_time_figure(lap_times_data)

    # ── 8. Stats bar ──────────────────────────────────────────────────────────
    @app.callback(
        Output("stats-bar", "style"),
        Output("chip-peak-stress-val", "children"),
        Output("chip-peak-stress-val", "style"),
        Output("chip-avg-stress-val", "children"),
        Output("chip-fastest-lap-val", "children"),
        Output("chip-pit-stops-val", "children"),
        Output("chip-radio-count-val", "children"),
        Input("store-lap-stress", "data"),
        prevent_initial_call=True,
    )
    def update_stats_bar(lap_stress_data):
        if not lap_stress_data or not lap_stress_data.get("stats"):
            raise PreventUpdate
        stats = lap_stress_data["stats"]
        base_val_style = {
            "color": WHITE,
            "fontSize": "16px",
            "fontWeight": "700",
            "fontFamily": FONT_MONO,
            "letterSpacing": "0.04em",
            "lineHeight": "1.2",
        }
        peak = stats["peak_stress"]
        peak_color = RED if peak >= 7.0 else (ORANGE if peak >= 5.0 else GREEN)
        stats_bar_style = {
            "backgroundColor": SURFACE,
            "borderBottom": f"1px solid {BORDER}",
            "padding": "10px 28px",
            "display": "flex",
            "gap": "10px",
            "flexWrap": "wrap",
            "alignItems": "stretch",
        }
        return (
            stats_bar_style,
            f"{peak:.1f}/10",
            {**base_val_style, "color": peak_color},
            f"{stats['avg_stress']:.1f}/10",
            stats["fastest_lap"],
            str(stats["pit_stops"]),
            str(stats["radio_count"]),
        )

    # ── 9. Show radio panel on marker click ──────────────────────────────────
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

        click_data = track_click or tel_click
        if not click_data:
            raise PreventUpdate

        points = click_data.get("points", [])
        if not points:
            raise PreventUpdate

        point = points[0]
        custom = point.get("customdata")
        if custom is None:
            raise PreventUpdate
        # Dash wraps scalar customdata in a list for list-based customdata traces
        if isinstance(custom, (list, tuple)):
            if not custom:
                raise PreventUpdate
            custom = custom[0]
        if not isinstance(custom, (int, float)):
            raise PreventUpdate

        idx = int(custom)
        # radio_records holds the merged all_radio list (D1 + D2), so validate
        # against its full length — D2 indices start at len(D1 records)
        if idx < 0 or idx >= len(radio_records):
            raise PreventUpdate

        rec = radio_records[idx]
        return _build_radio_panel(rec)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_fastest_lap_band(session, driver_code: str):
    """
    Return fastest lap timing as a dict for telemetry band overlay.
    Keys: start_iso, end_iso, lap_num, lap_time_s.
    Returns None on any failure.
    """
    try:
        lap = session.laps.pick_drivers(driver_code).pick_fastest()
        if lap is None:
            return None
        start = pd.Timestamp(lap["LapStartDate"])
        if hasattr(start, "tz") and start.tz is not None:
            start = start.tz_convert(None)
        lt = lap["LapTime"]
        if pd.isna(lt):
            return None
        return {
            "start_iso": start.isoformat(),
            "end_iso":   (start + lt).isoformat(),
            "lap_num":   int(lap["LapNumber"]),
            "lap_time_s": lt.total_seconds(),
        }
    except Exception:
        return None


def _add_fastest_lap_band(fig, band: dict):
    """Add a green vertical band across all 3 subplot rows for the fastest lap."""
    if not band:
        return
    x0, x1 = band["start_iso"], band["end_iso"]
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    for xref in ("x", "x2", "x3"):
        shapes.append(dict(
            type="rect",
            xref=xref, yref="paper",
            x0=x0, x1=x1,
            y0=0, y1=1,
            fillcolor="rgba(57,255,20,0.10)",
            line={"color": GREEN, "width": 1, "dash": "dot"},
            layer="below",
        ))
    fig.update_layout(shapes=shapes)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker={"color": GREEN, "size": 10, "symbol": "square"},
        name="Fastest Lap",
        showlegend=True,
    ), row=1, col=1)


def _add_pit_stop_markers(fig, pit_stops: list):
    """Add dashed white vertical lines + PIT labels on row 1 for each pit stop."""
    if not pit_stops:
        return
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    for stop in pit_stops:
        t_iso = stop["time"]
        shapes.append(dict(
            type="line",
            xref="x", yref="y domain",
            x0=t_iso, x1=t_iso,
            y0=0, y1=1,
            line={"color": "rgba(245,245,245,0.6)", "width": 1.5, "dash": "dash"},
        ))
    fig.update_layout(shapes=shapes)
    for stop in pit_stops:
        fig.add_annotation(
            x=stop["time"],
            y=1.0,
            xref="x",
            yref="paper",
            text=f"PIT L{stop['lap']}",
            showarrow=False,
            yanchor="top",
            font={"color": WHITE, "size": 8,
                  "family": "JetBrains Mono, Courier New, monospace"},
            opacity=0.75,
            bgcolor="rgba(22,22,22,0.7)",
            borderpad=2,
            xanchor="center",
        )
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line={"color": WHITE, "width": 1.5, "dash": "dash"},
        name="Pit Stop",
        showlegend=True,
    ), row=1, col=1)


def _add_drs_zones(fig, tel: pd.DataFrame):
    """
    Shade DRS activation zones on the top strip of the speed subplot (row 1).
    Derives zones from DRS==8 telemetry channel; consecutive samples are merged
    into bands (gap threshold 0.5 s). Fails silently on any error.
    """
    try:
        if "DRS" not in tel.columns or tel.empty:
            return
        drs_open = tel[tel["DRS"] == 8].copy()
        if drs_open.empty:
            return

        GAP = pd.Timedelta(seconds=0.5)
        bands = []
        band_start = prev_time = None
        for _, row in drs_open.iterrows():
            t = row["Date"]
            if band_start is None:
                band_start = t
            elif prev_time is not None and (t - prev_time) > GAP:
                bands.append({"start": band_start, "end": prev_time})
                band_start = t
            prev_time = t
        if band_start and prev_time:
            bands.append({"start": band_start, "end": prev_time})

        if not bands:
            return

        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        for band in bands:
            shapes.append(dict(
                type="rect",
                xref="x", yref="y domain",
                x0=band["start"].isoformat(),
                x1=band["end"].isoformat(),
                y0=0.84, y1=1.0,
                fillcolor="rgba(255,215,0,0.08)",
                line={"width": 0},
                layer="below",
            ))
        fig.update_layout(shapes=shapes)

        # Label only on first DRS zone to avoid clutter
        fig.add_annotation(
            x=bands[0]["start"].isoformat(),
            y=1.0,
            xref="x", yref="paper",
            text="DRS",
            showarrow=False,
            yanchor="top",
            font={"color": YELLOW, "size": 7,
                  "family": "JetBrains Mono, Courier New, monospace"},
            opacity=0.65,
            xanchor="left",
        )
    except Exception:
        pass


def _build_lap_time_figure(lap_times_data: list):
    """
    Line + scatter chart of lap time per lap.
    Compound-colored dots; SC laps = square, VSC = diamond, normal = circle.
    Y-axis formatted as M:SS.ss.
    """
    if not lap_times_data:
        from app.layout import _empty_figure
        return _empty_figure("")

    COMPOUND_COLORS = {
        "SOFT":    TYRE_SOFT,
        "MEDIUM":  TYRE_MEDIUM,
        "HARD":    TYRE_HARD,
        "INTER":   GREEN,
        "WET":     "#0078FF",
        "UNKNOWN": GREY,
    }

    laps_x    = [r["lap"]        for r in lap_times_data]
    times_y   = [r["lap_time_s"] for r in lap_times_data]
    compounds = [r["compound"]   for r in lap_times_data]

    dot_colors  = [COMPOUND_COLORS.get(c, GREY) for c in compounds]
    dot_symbols = []
    for r in lap_times_data:
        if r["is_sc"]:   dot_symbols.append("square")
        elif r["is_vsc"]: dot_symbols.append("diamond")
        else:             dot_symbols.append("circle")

    hover_texts = []
    for r in lap_times_data:
        lt = r["lap_time_s"]
        m, s = int(lt // 60), lt % 60
        flag = " [SC]" if r["is_sc"] else (" [VSC]" if r["is_vsc"] else "")
        hover_texts.append(f"Lap {r['lap']}: {m}:{s:06.3f}{flag}<br>{r['compound'].capitalize()}")

    fig = go.Figure()

    # Thin connector line
    fig.add_trace(go.Scatter(
        x=laps_x, y=times_y,
        mode="lines",
        line={"color": BORDER, "width": 1},
        hoverinfo="skip",
        showlegend=False,
    ))

    # Compound-colored dots
    fig.add_trace(go.Scatter(
        x=laps_x, y=times_y,
        mode="markers",
        marker={
            "color": dot_colors,
            "size": 7,
            "symbol": dot_symbols,
            "line": {"color": SURFACE, "width": 1},
        },
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    ))

    # Y-axis ticks as M:SS.ss
    if times_y:
        y_min = min(times_y) * 0.995
        y_max = max(times_y) * 1.005
        n_ticks = 5
        step = (y_max - y_min) / (n_ticks - 1)
        tick_vals = [y_min + i * step for i in range(n_ticks)]
        tick_texts = [f"{int(tv//60)}:{tv%60:05.2f}" for tv in tick_vals]
    else:
        y_min, y_max = 60, 120
        tick_vals, tick_texts = [], []

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": "JetBrains Mono, Courier New, monospace", "size": 10},
        margin={"l": 70, "r": 10, "t": 10, "b": 30},
        xaxis={
            "title": {"text": "Lap", "font": {"color": GREY, "size": 11}},
            "gridcolor": "#333333", "griddash": "dot", "zerolinecolor": BORDER,
            "tickfont": {"color": GREY, "size": 10},
            "dtick": 5,
        },
        yaxis={
            "title": {"text": "Lap Time", "font": {"color": GREY, "size": 11}},
            "gridcolor": "#333333", "griddash": "dot", "zerolinecolor": BORDER,
            "tickfont": {"color": GREY, "size": 10},
            "tickvals": tick_vals,
            "ticktext": tick_texts,
            "range": [y_min, y_max],
        },
        showlegend=False,
        hovermode="closest",
    )
    return fig


def _compute_lap_stress(tel: pd.DataFrame, stress: pd.Series, stint_data: pd.DataFrame) -> list:
    """
    Compute average stress per lap and return list of dicts for store-lap-stress.
    Each dict: {lap, avg_stress, compound, start_time, end_time}.
    """
    results = []
    if stint_data.empty or tel.empty:
        return results
    for _, row in stint_data.iterrows():
        try:
            start = pd.Timestamp(row["StintStart"])
            end = pd.Timestamp(row["StintEnd"])
            mask = (tel["Date"] >= start) & (tel["Date"] < end)
            avg = float(stress[mask].mean()) if mask.any() else 5.0
            if pd.isna(avg):
                avg = 5.0
            results.append({
                "lap": int(row["LapNumber"]),
                "avg_stress": round(avg, 2),
                "compound": str(row["Compound"]),
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
            })
        except Exception:
            continue
    return sorted(results, key=lambda r: r["lap"])


def _compute_leaderboard(session, model) -> list:
    """
    Compute stress metrics for every driver in the session.
    Uses anomaly z-score with coarser step for speed.
    Returns list sorted by avg_stress descending with rank assigned.
    """
    codes = get_all_driver_codes(session)
    if not codes:
        return []

    def _driver_stress(code):
        try:
            tel = get_telemetry(session, code)
            if tel.empty or len(tel) < 200:
                return None
            stress = predict_anomaly_zscore(tel, window=100, step=50)
            avg = float(stress.mean())
            max_s = float(stress.max())
            n = len(stress)
            step = max(1, n // 20)
            sparkline = [round(float(v), 2) for v in stress.iloc[::step].values[:20]]
            return {"code": code, "avg_stress": round(avg, 2), "max_stress": round(max_s, 2), "sparkline": sparkline}
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_driver_stress, c): c for c in codes}
        for future in futures:
            try:
                result = future.result()
            except Exception:
                result = None
            if result:
                results.append(result)

    results.sort(key=lambda r: r["avg_stress"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def _add_tyre_bands(fig, tel: pd.DataFrame, stint_data: pd.DataFrame):
    """
    Add tyre compound background bands to row 1 of the telemetry subplot.
    Modifies fig in-place.
    """
    if stint_data.empty or tel.empty:
        return

    compound_colors = {
        "SOFT":    "rgba(225,6,0,0.22)",
        "MEDIUM":  "rgba(255,242,0,0.18)",
        "HARD":    "rgba(245,245,245,0.13)",
        "INTER":   "rgba(57,255,20,0.15)",
        "WET":     "rgba(0,120,255,0.15)",
    }
    compound_legend_colors = {
        "SOFT":   TYRE_SOFT,
        "MEDIUM": TYRE_MEDIUM,
        "HARD":   TYRE_HARD,
        "INTER":  GREEN,
        "WET":    "#0078FF",
    }

    # Group consecutive laps with same compound into single band
    bands = []
    prev_compound = None
    band_start = None
    for _, row in stint_data.sort_values("LapNumber").iterrows():
        compound = row["Compound"]
        if compound != prev_compound:
            if prev_compound is not None and band_start is not None:
                bands.append({"compound": prev_compound, "start": band_start, "end": row["StintStart"]})
            prev_compound = compound
            band_start = row["StintStart"]
    if prev_compound and band_start is not None:
        bands.append({"compound": prev_compound, "start": band_start, "end": stint_data["StintEnd"].max()})

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    legend_compounds = set()

    for band in bands:
        color = compound_colors.get(band["compound"], "rgba(100,100,100,0.07)")
        shapes.append(dict(
            type="rect",
            xref="x", yref="y domain",
            x0=band["start"].isoformat() if hasattr(band["start"], "isoformat") else str(band["start"]),
            x1=band["end"].isoformat() if hasattr(band["end"], "isoformat") else str(band["end"]),
            y0=0, y1=1,
            fillcolor=color,
            line={"width": 0},
            layer="below",
        ))
        legend_compounds.add(band["compound"])

    fig.update_layout(shapes=shapes)

    # Dummy legend traces for tyre compounds
    for compound in sorted(legend_compounds):
        lc = compound_legend_colors.get(compound, GREY)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker={"color": lc, "size": 10, "symbol": "square"},
            name=f"{compound.capitalize()} tyre",
            showlegend=True,
        ), row=1, col=1)


def _add_incident_bands(fig, incidents: list):
    """
    Add SC/VSC period background bands across all three subplot rows.
    Modifies fig in-place.
    """
    if not incidents:
        return

    # Parse and sort by date
    events = []
    for item in incidents:
        try:
            dt = pd.to_datetime(item["date"])
            if hasattr(dt, "tz") and dt.tz is not None:
                dt = dt.tz_convert(None)
            events.append({**item, "_dt": dt})
        except Exception:
            continue
    events.sort(key=lambda e: e["_dt"])

    # Pair DEPLOYED → WITHDRAWN (or next DEPLOYED as proxy end)
    sc_periods = []
    vsc_periods = []
    i = 0
    while i < len(events):
        ev = events[i]
        msg  = (ev.get("message")  or "").upper()
        cat  = (ev.get("category") or "").upper()
        flag = (ev.get("flag")     or "").upper()
        is_sc = "SAFETY CAR" in msg and "VIRTUAL" not in msg
        is_vsc = "VIRTUAL SAFETY CAR" in msg or "VSC" in cat
        is_deployed = "DEPLOYED" in msg or "SAFETY CAR DEPLOYED" in msg or "SAFETY CAR OUT" in msg
        is_withdrawn = "IN THIS LAP" in msg or "WITHDRAWN" in msg or "ENDING" in msg

        if (is_sc or is_vsc) and is_deployed:
            # Find end: next withdrawn or next deployment
            end_dt = None
            for j in range(i + 1, len(events)):
                next_msg = events[j].get("message", "").upper()
                if "IN THIS LAP" in next_msg or "WITHDRAWN" in next_msg:
                    end_dt = events[j]["_dt"]
                    break
            if end_dt is None:
                # No explicit WITHDRAWN event found — cap the period at 15 minutes
                # to avoid a band spanning to the end of the race
                end_dt = ev["_dt"] + pd.Timedelta(minutes=15)
            if end_dt is None:
                i += 1
                continue
            period = {"start": ev["_dt"], "end": end_dt}
            if is_vsc:
                vsc_periods.append(period)
            else:
                sc_periods.append(period)
        i += 1

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []

    # 3 xref axes for the 3 subplot rows
    xrefs = ["x", "x2", "x3"]
    for period in sc_periods:
        x0 = period["start"].isoformat()
        x1 = period["end"].isoformat()
        for xref in xrefs:
            shapes.append(dict(
                type="rect",
                xref=xref, yref="paper",
                x0=x0, x1=x1,
                y0=0, y1=1,
                fillcolor="rgba(255,215,0,0.10)",
                line={"color": YELLOW, "width": 1, "dash": "dot"},
                layer="below",
            ))

    for period in vsc_periods:
        x0 = period["start"].isoformat()
        x1 = period["end"].isoformat()
        for xref in xrefs:
            shapes.append(dict(
                type="rect",
                xref=xref, yref="paper",
                x0=x0, x1=x1,
                y0=0, y1=1,
                fillcolor="rgba(255,107,53,0.10)",
                line={"color": ORANGE, "width": 1, "dash": "dot"},
                layer="below",
            ))

    fig.update_layout(shapes=shapes)

    # Legend entries
    if sc_periods:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker={"color": YELLOW, "size": 10, "symbol": "square"},
            name="Safety Car",
            showlegend=True,
        ), row=1, col=1)
    if vsc_periods:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker={"color": ORANGE, "size": 10, "symbol": "square"},
            name="VSC",
            showlegend=True,
        ), row=1, col=1)


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


def _build_telemetry_figure(
    tel, stress, radio_records, driver_code, stress_mode="RF Model",
    tel2=None, stress2=None, radio_records2=None, driver_code2=None,
    stint_data=None, incidents=None,
    pit_stops=None, fastest_lap_band=None,
):
    """
    Three-row subplot:
      Row 1 (50%): Speed + Brake dots [+ tyre bands + SC/VSC bands]
      Row 2 (30%): AI Stress [+ driver 2 stress overlay]
      Row 3 (20%): Throttle %
    Radio events appear as dashed lines + clickable diamonds on row 1.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.30, 0.20],
        vertical_spacing=0.03,
    )

    x = tel["Date"]

    # ── Row 1: Speed ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=tel["Speed"],
        name=f"Speed {driver_code} (km/h)",
        line={"color": WHITE, "width": 1.5},
        hovertemplate="Speed: %{y:.0f} km/h<extra></extra>",
    ), row=1, col=1)

    # Brake: render as a filled-under area (0 = no brake, speed value = braking)
    # This gives a clean "shade under the speed line when braking" effect instead
    # of thousands of dots obscuring the trace.
    brake_mask = tel["Brake"].astype(bool)
    brake_y = tel["Speed"].where(brake_mask, other=0)
    fig.add_trace(go.Scatter(
        x=x, y=brake_y,
        name="Braking",
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.35)",
        line={"color": "rgba(255,107,53,0.0)", "width": 0},
        hoverinfo="skip",
        showlegend=True,
    ), row=1, col=1)

    # Driver 2 speed overlay
    if tel2 is not None and not tel2.empty:
        fig.add_trace(go.Scatter(
            x=tel2["Date"], y=tel2["Speed"],
            name=f"Speed {driver_code2} (km/h)",
            line={"color": ORANGE, "width": 1.2},
            opacity=0.65,
            hovertemplate=f"{driver_code2} Speed: %{{y:.0f}} km/h<extra></extra>",
        ), row=1, col=1)

    # ── Row 2: AI Stress ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=stress,
        name=f"Stress {driver_code} [{stress_mode}]",
        fill="tozeroy",
        fillcolor="rgba(225,6,0,0.22)",
        line={"color": RED, "width": 1.5},
        hovertemplate="Stress: %{y:.2f}/10<extra></extra>",
    ), row=2, col=1)

    # Driver 2 stress overlay
    if stress2 is not None and tel2 is not None:
        fig.add_trace(go.Scatter(
            x=tel2["Date"], y=stress2,
            name=f"Stress {driver_code2}",
            fill="tozeroy",
            fillcolor="rgba(255,107,53,0.12)",
            line={"color": ORANGE, "width": 1.5},
            hovertemplate=f"{driver_code2} Stress: %{{y:.2f}}/10<extra></extra>",
        ), row=2, col=1)

    # ── Row 3: Throttle ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=tel["Throttle"],
        name="Throttle %",
        line={"color": GREEN, "width": 1.2},
        fill="tozeroy",
        fillcolor="rgba(57,255,20,0.12)",
        hovertemplate="Throttle: %{y:.0f}%<extra></extra>",
    ), row=3, col=1)

    # ── Radio markers: driver 1 ────────────────────────────────────────────────
    _add_radio_markers(fig, tel, stress, radio_records, RED, WHITE, offset=0)

    # ── Radio markers: driver 2 ────────────────────────────────────────────────
    if radio_records2 and tel2 is not None:
        _add_radio_markers(fig, tel2, stress2, radio_records2, ORANGE, WHITE, offset=len(radio_records))

    # ── Tyre bands + incident bands AFTER traces so xaxis.type is auto "date" ─
    fig.update_xaxes(type="date")
    if stint_data is not None and not stint_data.empty:
        _add_tyre_bands(fig, tel, stint_data)
    if incidents:
        _add_incident_bands(fig, incidents)
    # New feature overlays (must come after xaxis type is set)
    if fastest_lap_band:
        _add_fastest_lap_band(fig, fastest_lap_band)
    if pit_stops:
        _add_pit_stop_markers(fig, pit_stops)
    _add_drs_zones(fig, tel)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": "JetBrains Mono, Courier New, monospace", "size": 11},
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        legend={
            "bgcolor": "rgba(0,0,0,0.5)",
            "bordercolor": BORDER,
            "borderwidth": 1,
            "font": {"size": 10, "color": WHITE},
            "orientation": "h",
            "y": 1.04,
            "x": 0,
        },
        hovermode="x unified",
        dragmode="zoom",
    )

    fig.update_xaxes(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont={"color": GREY, "size": 10},
        showticklabels=False, row=1, col=1,
    )
    fig.update_xaxes(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont={"color": GREY, "size": 10},
        showticklabels=False, row=2, col=1,
    )
    fig.update_xaxes(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont={"color": GREY, "size": 10},
        showticklabels=True, row=3, col=1,
    )

    fig.update_yaxes(
        title={"text": "Speed (km/h)", "font": {"color": WHITE, "size": 11}},
        gridcolor="#333333", griddash="dot", zerolinecolor=BORDER,
        tickfont={"color": GREY, "size": 10},
        row=1, col=1,
    )
    fig.update_yaxes(
        title={"text": "Stress Score", "font": {"color": RED, "size": 11}},
        range=[0, 10],
        gridcolor="#333333", griddash="dot", zerolinecolor=BORDER,
        tickfont={"color": RED, "size": 10},
        row=2, col=1,
    )
    fig.update_yaxes(
        title={"text": "Throttle %", "font": {"color": GREEN, "size": 11}},
        range=[0, 105],
        gridcolor="#333333", griddash="dot", zerolinecolor=BORDER,
        tickfont={"color": GREEN, "size": 10},
        row=3, col=1,
    )

    return fig


def _add_radio_markers(fig, tel, stress, radio_records, marker_color, text_color, offset=0):
    """Add dashed event lines + clickable diamonds for a set of radio records."""
    for i, rec in enumerate(radio_records):
        rt = pd.to_datetime(rec["time"])
        transcript_short = rec["transcript"][:45] + ("…" if len(rec["transcript"]) > 45 else "")

        # Thin dashed vertical line across all 3 rows using vline-style scatter
        for row_num, y_col in ((1, tel["Speed"]), (2, stress), (3, tel["Throttle"])):
            fig.add_trace(go.Scatter(
                x=[rt, rt],
                y=[float(y_col.min()), float(y_col.max())],
                mode="lines",
                line={"color": marker_color, "width": 0.8, "dash": "dot"},
                opacity=0.30,
                hoverinfo="skip",
                showlegend=False,
            ), row=row_num, col=1)

        closest_idx = (tel["Date"] - rt).abs().idxmin()
        y_speed = float(tel["Speed"].loc[closest_idx])
        code = rec.get("code", "")
        label = f"{code[:1]}{i+1}" if code else f"R{i+1}"

        fig.add_trace(go.Scatter(
            x=[rt],
            y=[y_speed],
            mode="markers+text",
            marker={"color": marker_color, "size": 10, "symbol": "diamond",
                    "line": {"color": text_color, "width": 1.2}},
            text=[label],
            textposition="top center",
            textfont={"color": text_color, "size": 8},
            customdata=[offset + i],
            name="Radio",
            hovertemplate=(
                f"<b>Radio #{i+1} ({rec.get('code', '')})</b><br>"
                f"{transcript_short}<br>"
                f"Stress: {rec['stress']:.1f}/10"
                "<extra></extra>"
            ),
            showlegend=False,
        ), row=1, col=1)


def _build_track_figure(
    pos, radio_records,
    pos2=None, radio_records2=None, driver_code2=None,
    incidents=None,
):
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
        showlegend=True,
        legend={
            "bgcolor": "rgba(0,0,0,0.5)",
            "bordercolor": BORDER,
            "font": {"size": 9, "color": WHITE},
            "x": 0.01, "y": 0.99,
        },
    )

    # ── Driver 2 track (drawn first, underneath) ──────────────────────────────
    if pos2 is not None and not pos2.empty:
        fig.add_trace(go.Scatter(
            x=pos2["X"], y=pos2["Y"],
            mode="lines",
            line={"color": "#3A2A1A", "width": 8},
            hoverinfo="skip",
            name=f"Track {driver_code2}",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=pos2["X"], y=pos2["Y"],
            mode="lines",
            line={"color": "#664422", "width": 2},
            hoverinfo="skip",
            name=f"Centre {driver_code2}",
            showlegend=False,
        ))

    # ── Driver 1 track ────────────────────────────────────────────────────────
    if not pos.empty:
        fig.add_trace(go.Scatter(
            x=pos["X"], y=pos["Y"],
            mode="lines",
            line={"color": "#3A3A3A", "width": 8},
            hoverinfo="skip",
            name="Track",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=pos["X"], y=pos["Y"],
            mode="lines",
            line={"color": "#555555", "width": 2},
            hoverinfo="skip",
            name="Centre line",
            showlegend=False,
        ))

    # ── SC/VSC legend entries on track map ────────────────────────────────────
    if incidents:
        has_sc = any("VIRTUAL" not in str(e.get("message", "")).upper()
                     and "SAFETY CAR" in str(e.get("message", "")).upper()
                     for e in incidents)
        has_vsc = any("VIRTUAL SAFETY CAR" in str(e.get("message", "")).upper()
                      for e in incidents)
        if has_sc:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker={"color": YELLOW, "size": 10, "symbol": "square"},
                name="Safety Car",
            ))
        if has_vsc:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker={"color": ORANGE, "size": 10, "symbol": "square"},
                name="VSC",
            ))

    # ── Driver 1 radio markers ────────────────────────────────────────────────
    _add_track_radio_markers(fig, radio_records, RED, WHITE, offset=0)

    # ── Driver 2 radio markers ────────────────────────────────────────────────
    if radio_records2 and pos2 is not None:
        _add_track_radio_markers(fig, radio_records2, ORANGE, WHITE, offset=len(radio_records))

    return fig


def _add_track_radio_markers(fig, radio_records, marker_color, text_color, offset=0):
    """Add radio event dots on the track map for a set of records."""
    valid = [(i, r) for i, r in enumerate(radio_records)
             if r.get("x") is not None and r.get("y") is not None]
    if not valid:
        return

    xs = [r["x"] for _, r in valid]
    ys = [r["y"] for _, r in valid]
    indices = [offset + i for i, _ in valid]
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
            "line": {"color": text_color, "width": 1.5},
        },
        customdata=indices,
        hovertext=[f"{t}  |  Stress {s:.1f}/10<br>{tr}" for t, s, tr in zip(times, stresses, transcripts)],
        hoverinfo="text",
        name="Radio",
        showlegend=False,
    ))

    # Pulse ring
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker={
            "color": "rgba(0,0,0,0)",
            "size": 22,
            "symbol": "circle-open",
            "line": {"color": marker_color, "width": 1},
            "opacity": 0.4,
        },
        customdata=indices,
        hoverinfo="skip",
        showlegend=False,
    ))


def _stress_colour(stress: float) -> str:
    """Map a stress score 1–10 to a colour between green and red."""
    t = (stress - 1.0) / 9.0
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
    driver_label = rec.get("code", rec.get("driver", ""))
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
            html.Div(style={"minWidth": "180px"}, children=[
                html.Div(
                    f"RADIO  ·  {driver_label}" if driver_label else "RADIO TRANSMISSION",
                    style={**LABEL_STYLE, "marginBottom": "8px"},
                ),
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

            html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                html.Div("TRANSCRIPT", style={**LABEL_STYLE, "marginBottom": "8px"}),
                html.Div(
                    f'"{transcript}"',
                    style={
                        "fontSize": "14px", "lineHeight": "1.6",
                        "fontStyle": "italic" if transcript != "[engine static]" else "normal",
                        "color": GREY if transcript == "[engine static]" else WHITE,
                    },
                ),
            ]),

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


def _build_leaderboard_panel(leaderboard_data: list):
    """Build the driver stress leaderboard HTML component."""
    rows = []
    for entry in leaderboard_data:
        rank = entry["rank"]
        code = entry["code"]
        avg = entry["avg_stress"]
        max_s = entry["max_stress"]
        bar_pct = int((avg / 10) * 100)
        stress_color = _stress_colour(avg)

        rank_color = {1: YELLOW, 2: WHITE, 3: ORANGE}.get(rank, GREY)

        rows.append(html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "16px",
                "padding": "8px 0",
                "borderBottom": f"1px solid {BORDER}",
            },
            children=[
                # Rank
                html.Span(f"P{rank}", style={
                    "color": rank_color, "fontSize": "11px", "fontWeight": "700",
                    "minWidth": "28px", "letterSpacing": "0.05em",
                }),
                # Driver code
                html.Span(code, style={
                    "color": WHITE, "fontSize": "13px", "fontWeight": "700",
                    "minWidth": "40px", "letterSpacing": "0.1em",
                }),
                # Stress bar
                html.Div(style={"flex": "1", "minWidth": "80px", "maxWidth": "200px"}, children=[
                    html.Div(style={
                        "height": "4px", "width": "100%",
                        "backgroundColor": BORDER, "borderRadius": "2px",
                    }, children=[
                        html.Div(style={
                            "height": "4px", "width": f"{bar_pct}%",
                            "backgroundColor": stress_color, "borderRadius": "2px",
                        })
                    ]),
                ]),
                # Avg stress value
                html.Span(f"{avg:.1f}", style={
                    "color": stress_color, "fontSize": "12px", "fontWeight": "700",
                    "minWidth": "30px",
                }),
                # Max stress
                html.Span(f"max {max_s:.1f}", style={
                    "color": GREY, "fontSize": "10px", "letterSpacing": "0.05em",
                }),
            ],
        ))

    return html.Div(
        style={
            "backgroundColor": CARD,
            "border": f"1px solid {BORDER}",
            "borderRadius": "6px",
            "padding": "16px 24px",
        },
        children=[
            html.Div("DRIVER STRESS LEADERBOARD", style={**LABEL_STYLE, "marginBottom": "12px"}),
            html.Div(rows),
        ],
    )
