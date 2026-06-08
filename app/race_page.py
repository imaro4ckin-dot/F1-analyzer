"""
Race Overview Page
──────────────────
Shows a full picture of the selected race:
  • Race header (circuit, date, winner)
  • Results table  (P, Driver, Team, Gap, Fastest Lap, Pits, Status)
  • Lap-by-lap position chart for every driver
  • Tyre strategy timeline
  • Fastest lap comparison bar chart
  • Race incidents / flag timeline
  • Interactive Race Map: slider + play/pause to watch all cars move around the track
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from app.layout import (
    BG, SURFACE, CARD, CARD2, BORDER, BORDER2,
    RED, RED_GLOW, WHITE, MUTED, MUTED2,
    GREEN, ORANGE, YELLOW, TEAL,
    TYRE_SOFT, TYRE_MEDIUM, TYRE_HARD,
    FONT_MONO, FONT_SANS,
    LABEL_STYLE, _card, _section_header, _empty_figure,
)

# ── Tyre palette ──────────────────────────────────────────────────────────────
_COMPOUND_COLOR = {
    "SOFT":    TYRE_SOFT,
    "MEDIUM":  TYRE_MEDIUM,
    "HARD":    TYRE_HARD,
    "INTER":   GREEN,
    "WET":     "#0078FF",
    "UNKNOWN": MUTED2,
}

# Team → brand color (best-effort fallback to MUTED)
_TEAM_COLOR = {
    "Red Bull Racing":   "#3671C6",
    "Ferrari":           "#E8000D",
    "Mercedes":          "#27F4D2",
    "McLaren":           "#FF8000",
    "Aston Martin":      "#358C75",
    "Alpine":            "#FF87BC",
    "Williams":          "#64C4FF",
    "RB":                "#6692FF",
    "Kick Sauber":       "#52E252",
    "Haas F1 Team":      "#B6BABD",
}


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

def build_race_layout():
    """Returns the Race Overview page content div."""
    return html.Div(children=[

        # ── Hidden store (kept for potential future use, not needed for animation) ─
        dcc.Store(id="store-race-map-data"),


        # ── Race controls bar ─────────────────────────────────────────────
        html.Div(
            id="race-controls-panel",
            style={
                "backgroundColor": SURFACE,
                "borderBottom": f"1px solid {BORDER}",
                "padding": "14px 32px",
                "display": "flex",
                "gap": "14px",
                "alignItems": "flex-end",
                "flexWrap": "wrap",
            },
            children=[
                html.Div([
                    html.Div("Season", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="race-dd-year",
                        options=[{"label": str(y), "value": y} for y in [2023, 2024, 2025]],
                        value=2024,
                        clearable=False,
                        style={
                            "backgroundColor": CARD2,
                            "color": RED,
                            "border": f"1px solid {BORDER2}",
                            "borderRadius": "8px",
                            "fontFamily": FONT_MONO,
                            "fontSize": "12px",
                        },
                        className="f1-dropdown",
                    ),
                ], style={"width": "100px"}),

                html.Div([
                    html.Div("Grand Prix", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="race-dd-race",
                        options=[],
                        placeholder="Select race…",
                        clearable=False,
                        style={
                            "backgroundColor": CARD2,
                            "color": RED,
                            "border": f"1px solid {BORDER2}",
                            "borderRadius": "8px",
                            "fontFamily": FONT_MONO,
                            "fontSize": "12px",
                        },
                        className="f1-dropdown",
                    ),
                ], style={"minWidth": "220px", "flex": "2"}),

                html.Div([
                    html.Div("\u00a0", style={**LABEL_STYLE, "visibility": "hidden"}),
                    html.Button(
                        id="race-btn-load",
                        n_clicks=0,
                        className="btn-analyze",
                        children=html.Span("LOAD RACE", style={
                            "letterSpacing": "0.2em",
                            "fontSize": "11px",
                            "fontWeight": "700",
                            "fontFamily": FONT_SANS,
                        }),
                        style={
                            "background": f"linear-gradient(135deg, {RED} 0%, #C8000A 100%)",
                            "color": WHITE,
                            "border": "none",
                            "padding": "0 28px",
                            "height": "38px",
                            "cursor": "pointer",
                            "borderRadius": "8px",
                            "transition": "all 0.2s",
                            "boxShadow": f"0 4px 20px {RED_GLOW}",
                            "outline": "none",
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                ]),

                # Status message (right-aligned)
                html.Div(
                    id="race-status-msg",
                    style={
                        "marginLeft": "auto",
                        "color": MUTED,
                        "fontSize": "10px",
                        "letterSpacing": "0.1em",
                        "fontFamily": FONT_MONO,
                        "textAlign": "right",
                        "alignSelf": "flex-end",
                        "paddingBottom": "2px",
                    },
                ),
            ],
        ),

        # ── Main race content ─────────────────────────────────────────────
        html.Main(
            style={"padding": "24px 32px 48px 32px"},
            children=[

                # Race header banner
                html.Div(
                    id="race-header-banner",
                    style={"marginBottom": "20px"},
                ),

                # Row 1: Results table + Fastest laps chart
                html.Div(
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
                    children=[
                        _card(
                            flex="5", min_width="400px",
                            children=[
                                _section_header("Race Results", badge="final classification"),
                                html.Div(id="race-results-table"),
                            ],
                        ),
                        _card(
                            flex="3", min_width="280px",
                            children=[
                                _section_header("Fastest Laps", badge="per driver"),
                                dcc.Graph(
                                    id="race-chart-fastest-laps",
                                    config={"displayModeBar": False, "displaylogo": False},
                                    style={"height": "420px"},
                                    figure=_empty_figure(""),
                                ),
                            ],
                        ),
                    ],
                ),

                # Row 2: Position chart (full width)
                _card(
                    extra_style={"marginBottom": "16px"},
                    children=[
                        _section_header(
                            "Lap-by-Lap Positions",
                            badge="all drivers · click legend to isolate",
                        ),
                        dcc.Graph(
                            id="race-chart-positions",
                            config={
                                "displayModeBar": True,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"],
                                "displaylogo": False,
                            },
                            style={"height": "420px"},
                            figure=_empty_figure("Load a race to see positions"),
                        ),
                    ],
                ),

                # Row 3: Tyre strategy (full width)
                _card(
                    extra_style={"marginBottom": "16px"},
                    children=[
                        _section_header(
                            "Tyre Strategy",
                            badge="stint breakdown · all drivers",
                        ),
                        dcc.Graph(
                            id="race-chart-tyres",
                            config={"displayModeBar": False, "displaylogo": False},
                            style={"height": "420px"},
                            figure=_empty_figure(""),
                        ),
                    ],
                ),

                # Row 4: Incidents timeline (full width)
                html.Div(
                    id="race-incidents-panel",
                    style={"marginBottom": "16px"},
                ),

                # Row 5: Race Map (full width, initially hidden)
                build_race_map_section(),
            ],
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def register_race_callbacks(app):

    # 1. Populate race dropdown when year changes
    @app.callback(
        Output("race-dd-race", "options"),
        Output("race-dd-race", "value"),
        Input("race-dd-year", "value"),
    )
    def race_update_races(year):
        if not year:
            raise PreventUpdate
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
            from core.data_loader import fetch_races
            races = fetch_races(int(year))
            options = [
                {"label": f"{r['country_name']}  ·  {r['location']}", "value": r["location"]}
                for r in races
            ]

        value = options[0]["value"] if options else None
        return options, value

    # 2. Load race data and render all charts
    @app.callback(
        Output("race-header-banner",       "children"),
        Output("race-results-table",       "children"),
        Output("race-chart-positions",     "figure"),
        Output("race-chart-tyres",         "figure"),
        Output("race-chart-fastest-laps",  "figure"),
        Output("race-incidents-panel",     "children"),
        Output("race-status-msg",          "children"),
        Output("race-map-graph",           "figure"),
        Output("race-map-section",         "style"),
        Input("race-btn-load",             "n_clicks"),
        State("race-dd-year",              "value"),
        State("race-dd-race",              "value"),
        prevent_initial_call=True,
    )
    def load_race(n_clicks, year, location):
        if not year or not location:
            raise PreventUpdate

        hidden = {"display": "none"}
        shown  = {"display": "block", "marginTop": "0"}

        try:
            from core.data_loader import load_session
            session = load_session(int(year), location)
        except Exception as e:
            msg = f"Session load failed: {e}"
            empty = _empty_figure("Error loading session")
            return [], html.Div(msg, style={"color": RED}), empty, empty, empty, [], msg, empty, hidden

        # ── Pull session results ──────────────────────────────────────────
        try:
            results = session.results
        except Exception:
            results = pd.DataFrame()

        # ── Race control events ───────────────────────────────────────────
        try:
            session_key = int(session.session_info["Key"])
            import requests
            rc_all = requests.get(
                f"https://api.openf1.org/v1/race_control?session_key={session_key}",
                timeout=10,
            ).json()
            if not isinstance(rc_all, list):
                rc_all = []
        except Exception:
            rc_all = []

        # ── Lap-by-lap positions ──────────────────────────────────────────
        try:
            all_laps = session.laps[["DriverNumber", "LapNumber", "Position", "LapTime"]].copy()
            all_laps = all_laps[all_laps["Position"].notna()]
        except Exception:
            all_laps = pd.DataFrame()

        # ── Tyre stints ───────────────────────────────────────────────────
        try:
            stints_df = session.laps[[
                "DriverNumber", "LapNumber", "Compound", "TyreLife", "FreshTyre"
            ]].copy()
        except Exception:
            stints_df = pd.DataFrame()

        # ── Build components ──────────────────────────────────────────────
        header    = _build_race_header(session, results)
        tbl       = _build_results_table(results, session)
        pos_fig   = _build_positions_chart(all_laps, session)
        tyre_fig  = _build_tyre_strategy(stints_df, results, session)
        fl_fig    = _build_fastest_laps_chart(results, session)
        incidents = _build_incidents_panel(rc_all)

        driver_count = len(results) if not results.empty else "?"
        lap_count = int(all_laps["LapNumber"].max()) if not all_laps.empty else "?"
        status = f"{driver_count} drivers  ·  {lap_count} laps  ·  {location} {year}"

        # ── Animated race map (all frames embedded — pure client-side) ────
        try:
            map_payload = _build_map_frames(session)
            map_fig = _build_animated_figure(map_payload)
            map_style = shown
        except Exception:
            map_fig = _empty_figure("Map data unavailable")
            map_style = hidden

        return header, tbl, pos_fig, tyre_fig, fl_fig, incidents, status, map_fig, map_style



# ─────────────────────────────────────────────────────────────────────────────
# Component builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_race_header(session, results):
    """Banner showing circuit name, date, and winner."""
    try:
        event = session.event
        circuit = str(event.get("EventName", "Grand Prix"))
        country = str(event.get("Country", ""))
        date_raw = event.get("EventDate", None)
        date_str = pd.Timestamp(date_raw).strftime("%d %b %Y") if date_raw else ""
    except Exception:
        circuit, country, date_str = "Grand Prix", "", ""

    winner_name = "—"
    winner_team = ""
    if not results.empty:
        try:
            p1 = results[results["Position"] == 1].iloc[0]
            winner_name = str(p1.get("FullName") or p1.get("Abbreviation") or "—")
            winner_team = str(p1.get("TeamName", ""))
        except Exception:
            pass

    team_col = _TEAM_COLOR.get(winner_team, RED)

    return html.Div(
        style={
            "backgroundColor": CARD,
            "border": f"1px solid {BORDER}",
            "borderLeft": f"4px solid {RED}",
            "borderRadius": "16px",
            "padding": "20px 28px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "flexWrap": "wrap",
            "gap": "16px",
        },
        children=[
            # Left: circuit + date
            html.Div(children=[
                html.Div(country.upper(), style={
                    "color": RED,
                    "fontSize": "9px",
                    "letterSpacing": "0.3em",
                    "fontFamily": FONT_MONO,
                    "fontWeight": "600",
                    "marginBottom": "4px",
                }),
                html.Div(circuit, style={
                    "color": WHITE,
                    "fontSize": "22px",
                    "fontWeight": "700",
                    "fontFamily": FONT_SANS,
                    "letterSpacing": "0.04em",
                    "lineHeight": "1",
                    "marginBottom": "6px",
                }),
                html.Div(date_str, style={
                    "color": MUTED,
                    "fontSize": "11px",
                    "fontFamily": FONT_MONO,
                    "letterSpacing": "0.1em",
                }),
            ]),
            # Right: winner
            html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-end",
                    "gap": "4px",
                },
                children=[
                    html.Div("RACE WINNER", style={
                        "color": MUTED,
                        "fontSize": "8px",
                        "letterSpacing": "0.25em",
                        "fontFamily": FONT_MONO,
                        "fontWeight": "600",
                    }),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                        children=[
                            html.Div(style={
                                "width": "4px", "height": "28px",
                                "backgroundColor": team_col,
                                "borderRadius": "2px",
                            }),
                            html.Div(winner_name, style={
                                "color": WHITE,
                                "fontSize": "20px",
                                "fontWeight": "700",
                                "fontFamily": FONT_SANS,
                                "letterSpacing": "0.06em",
                            }),
                        ],
                    ),
                    html.Div(winner_team, style={
                        "color": MUTED,
                        "fontSize": "10px",
                        "fontFamily": FONT_MONO,
                        "letterSpacing": "0.08em",
                    }),
                ],
            ),
        ],
    )


def _build_results_table(results, session):
    """Full race classification table."""
    if results is None or (hasattr(results, "empty") and results.empty):
        return html.Div("No results available.", style={"color": MUTED, "fontSize": "12px", "fontFamily": FONT_MONO})

    # Header row
    col_style = {
        "color": MUTED,
        "fontSize": "8px",
        "letterSpacing": "0.22em",
        "fontFamily": FONT_MONO,
        "fontWeight": "600",
        "textTransform": "uppercase",
        "padding": "0 8px 10px 0",
    }
    header = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "36px 52px 1fr 1fr 90px 52px 44px 70px",
            "borderBottom": f"1px solid {BORDER2}",
            "paddingBottom": "6px",
            "marginBottom": "4px",
        },
        children=[
            html.Div("Pos",      style=col_style),
            html.Div("Driver",   style=col_style),
            html.Div("Name",     style=col_style),
            html.Div("Team",     style=col_style),
            html.Div("Gap",      style=col_style),
            html.Div("F.Lap",    style=col_style),
            html.Div("Pits",     style=col_style),
            html.Div("Status",   style=col_style),
        ],
    )

    rows = [header]
    try:
        sorted_res = results.sort_values("Position")
    except Exception:
        sorted_res = results

    for _, row in sorted_res.iterrows():
        pos = row.get("Position", "—")
        try:
            pos = int(pos)
        except Exception:
            pos = "—"

        abbr      = str(row.get("Abbreviation", "—"))
        full_name = str(row.get("FullName", abbr))
        team      = str(row.get("TeamName", "—"))
        status    = str(row.get("Status", "—"))
        points    = row.get("Points", None)

        # Gap to leader
        try:
            gap_val = row.get("Time", None) or row.get("Gap", None)
            if pd.isna(gap_val):
                gap_str = status if status not in ("Finished",) else "—"
            else:
                gap_s = gap_val.total_seconds() if hasattr(gap_val, "total_seconds") else float(gap_val)
                if gap_s < 0.001:
                    gap_str = "WINNER"
                elif gap_s > 600:
                    gap_str = f"+{int(gap_s//60)} laps"
                else:
                    gap_str = f"+{gap_s:.3f}s"
        except Exception:
            gap_str = "—"

        # Fastest lap flag
        try:
            fl = bool(row.get("FastestLap", False))
        except Exception:
            fl = False

        # Pit stop count from session laps
        try:
            driver_num = row.get("DriverNumber") or row.get("BroadcastName", "")
            pit_count = int(session.laps[
                session.laps["DriverNumber"] == str(driver_num)
            ]["PitInTime"].notna().sum())
        except Exception:
            pit_count = "—"

        # Position colors
        pos_color = {1: YELLOW, 2: WHITE, 3: ORANGE}.get(pos, MUTED)
        team_color = _TEAM_COLOR.get(team, MUTED2)

        is_dnf = status not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps")

        cell = {
            "color": WHITE if not is_dnf else MUTED,
            "fontSize": "12px",
            "fontFamily": FONT_MONO,
            "padding": "8px 8px 8px 0",
            "letterSpacing": "0.04em",
            "display": "flex",
            "alignItems": "center",
        }

        row_div = html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "36px 52px 1fr 1fr 90px 52px 44px 70px",
                "borderBottom": f"1px solid {BORDER}",
                "alignItems": "center",
                "transition": "background 0.15s",
            },
            className="lb-row",
            children=[
                # Position
                html.Div(str(pos), style={**cell, "color": pos_color, "fontWeight": "700"}),
                # Abbreviation
                html.Div(
                    style={**cell, "gap": "6px"},
                    children=[
                        html.Div(style={
                            "width": "3px", "height": "16px",
                            "backgroundColor": team_color,
                            "borderRadius": "2px",
                            "flexShrink": "0",
                        }),
                        html.Span(abbr, style={"fontWeight": "700", "letterSpacing": "0.10em"}),
                    ],
                ),
                # Full name
                html.Div(full_name, style={**cell, "fontSize": "11px", "color": WHITE if not is_dnf else MUTED}),
                # Team
                html.Div(team, style={**cell, "fontSize": "10px", "color": MUTED}),
                # Gap
                html.Div(gap_str, style={
                    **cell,
                    "color": YELLOW if gap_str == "WINNER" else (MUTED if gap_str == "—" else WHITE),
                    "fontWeight": "600" if gap_str == "WINNER" else "400",
                }),
                # Fastest lap indicator
                html.Div(
                    "●" if fl else "",
                    style={**cell, "color": TEAL, "fontSize": "14px", "justifyContent": "center"},
                ),
                # Pit count
                html.Div(str(pit_count), style={**cell, "color": MUTED, "justifyContent": "center"}),
                # Status
                html.Div(
                    status if is_dnf else "Finished",
                    style={
                        **cell,
                        "fontSize": "10px",
                        "color": RED if is_dnf else GREEN,
                    },
                ),
            ],
        )
        rows.append(row_div)

    return html.Div(
        style={"overflowY": "auto", "maxHeight": "420px"},
        children=rows,
    )


def _build_positions_chart(all_laps, session):
    """Line chart: lap number vs position for every driver."""
    if all_laps.empty:
        return _empty_figure("No lap position data available")

    fig = go.Figure()

    # Get driver abbreviation map
    driver_abbr = {}
    try:
        for num in session.drivers:
            info = session.get_driver(num)
            driver_abbr[str(num)] = info.get("Abbreviation", str(num))
    except Exception:
        pass

    # Group by driver, sorted by final position
    try:
        final_pos = (
            all_laps.sort_values("LapNumber")
            .groupby("DriverNumber")
            .last()["Position"]
            .sort_values()
        )
        driver_order = final_pos.index.tolist()
    except Exception:
        driver_order = all_laps["DriverNumber"].unique().tolist()

    for drv_num in driver_order:
        drv_laps = all_laps[all_laps["DriverNumber"] == drv_num].sort_values("LapNumber")
        if drv_laps.empty:
            continue
        abbr = driver_abbr.get(str(drv_num), str(drv_num))
        try:
            team = session.get_driver(str(drv_num)).get("TeamName", "")
        except Exception:
            team = ""
        color = _TEAM_COLOR.get(team, "#555577")

        fig.add_trace(go.Scatter(
            x=drv_laps["LapNumber"],
            y=drv_laps["Position"],
            mode="lines",
            name=abbr,
            line={"color": color, "width": 2},
            hovertemplate=f"<b>{abbr}</b>  Lap %{{x}}  P%{{y}}<extra></extra>",
        ))

    total_laps = int(all_laps["LapNumber"].max())
    num_drivers = len(driver_order)

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": FONT_MONO, "size": 10},
        margin={"l": 50, "r": 120, "t": 16, "b": 40},
        xaxis={
            "title": {"text": "Lap", "font": {"color": MUTED, "size": 11}},
            "gridcolor": "rgba(255,255,255,0.05)",
            "zerolinecolor": "rgba(255,255,255,0.05)",
            "tickfont": {"color": MUTED, "size": 10},
            "range": [1, total_laps],
        },
        yaxis={
            "title": {"text": "Position", "font": {"color": MUTED, "size": 11}},
            "gridcolor": "rgba(255,255,255,0.05)",
            "zerolinecolor": "rgba(255,255,255,0.05)",
            "tickfont": {"color": MUTED, "size": 10},
            "autorange": "reversed",
            "range": [num_drivers + 0.5, 0.5],
            "dtick": 1,
        },
        hovermode="x unified",
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 10, "color": WHITE},
            "orientation": "v",
            "x": 1.01, "y": 1,
            "xanchor": "left",
        },
        dragmode="zoom",
    )
    return fig


def _build_tyre_strategy(stints_df, results, session):
    """Horizontal bar chart showing tyre stints per driver."""
    if stints_df.empty:
        return _empty_figure("No tyre stint data available")

    # Build driver order by final position
    driver_abbr = {}
    driver_team = {}
    try:
        for num in session.drivers:
            info = session.get_driver(num)
            driver_abbr[str(num)] = info.get("Abbreviation", str(num))
            driver_team[str(num)] = info.get("TeamName", "")
    except Exception:
        pass

    if not results.empty:
        try:
            sorted_res = results.sort_values("Position")
            driver_order = [str(r["DriverNumber"]) for _, r in sorted_res.iterrows()
                            if str(r.get("DriverNumber", "")) in driver_abbr]
        except Exception:
            driver_order = list(driver_abbr.keys())
    else:
        driver_order = list(driver_abbr.keys())

    # Build stint blocks: consecutive laps with same compound per driver
    def _get_stints(drv_num):
        drv = stints_df[stints_df["DriverNumber"] == drv_num].sort_values("LapNumber")
        if drv.empty:
            return []
        stints = []
        start_lap = None
        cur_compound = None
        for _, r in drv.iterrows():
            c = str(r.get("Compound", "UNKNOWN") or "UNKNOWN").upper()
            lap = int(r["LapNumber"])
            if c != cur_compound:
                if cur_compound is not None:
                    stints.append({"start": start_lap, "end": lap - 1, "compound": cur_compound})
                cur_compound = c
                start_lap = lap
        if cur_compound:
            stints.append({"start": start_lap, "end": int(drv["LapNumber"].max()), "compound": cur_compound})
        return stints

    fig = go.Figure()
    y_labels = []
    already_in_legend = set()

    for drv_num in driver_order:
        abbr = driver_abbr.get(str(drv_num), str(drv_num))
        y_labels.append(abbr)
        stints = _get_stints(str(drv_num))
        for stint in stints:
            compound = stint["compound"]
            color = _COMPOUND_COLOR.get(compound, MUTED2)
            span = stint["end"] - stint["start"] + 1
            show_legend = compound not in already_in_legend
            if show_legend:
                already_in_legend.add(compound)
            fig.add_trace(go.Bar(
                x=[span],
                y=[abbr],
                base=[stint["start"] - 1],
                orientation="h",
                name=compound.capitalize(),
                legendgroup=compound,
                showlegend=show_legend,
                marker={
                    "color": color,
                    "line": {"color": CARD, "width": 1},
                },
                hovertemplate=(
                    f"<b>{abbr}</b> — {compound.capitalize()}<br>"
                    f"Laps {stint['start']}–{stint['end']}  ({span} laps)"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": FONT_MONO, "size": 10},
        margin={"l": 50, "r": 20, "t": 16, "b": 40},
        barmode="stack",
        xaxis={
            "title": {"text": "Lap", "font": {"color": MUTED, "size": 11}},
            "gridcolor": "rgba(255,255,255,0.05)",
            "zerolinecolor": "rgba(255,255,255,0.05)",
            "tickfont": {"color": MUTED, "size": 10},
        },
        yaxis={
            "tickfont": {"color": WHITE, "size": 10},
            "gridcolor": "rgba(255,255,255,0.03)",
            "autorange": "reversed",
        },
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 10, "color": WHITE},
            "orientation": "h",
            "x": 0, "y": 1.04,
        },
        bargap=0.25,
    )
    return fig


def _build_fastest_laps_chart(results, session):
    """Horizontal bar chart of fastest lap time per driver."""
    if results is None or (hasattr(results, "empty") and results.empty):
        return _empty_figure("No results available")

    rows = []
    try:
        for _, r in results.sort_values("Position").iterrows():
            drv = str(r.get("Abbreviation", "—"))
            team = str(r.get("TeamName", ""))
            try:
                drv_laps = session.laps.pick_drivers(drv)
                fl = drv_laps["LapTime"].dropna().min()
                fl_s = fl.total_seconds()
            except Exception:
                continue
            if fl_s <= 0 or fl_s > 300:
                continue
            rows.append({"driver": drv, "fl_s": fl_s, "team": team})
    except Exception:
        return _empty_figure("Error computing fastest laps")

    if not rows:
        return _empty_figure("No fastest lap data")

    rows.sort(key=lambda x: x["fl_s"])
    best = rows[0]["fl_s"]

    drivers   = [r["driver"] for r in rows]
    deltas    = [r["fl_s"] - best for r in rows]
    colors    = [_TEAM_COLOR.get(r["team"], MUTED2) for r in rows]
    hover     = [
        f"<b>{r['driver']}</b><br>"
        f"{int(r['fl_s']//60)}:{r['fl_s']%60:06.3f}<br>"
        f"+{r['fl_s']-best:.3f}s"
        for r in rows
    ]

    fig = go.Figure(go.Bar(
        x=deltas,
        y=drivers,
        orientation="h",
        marker={"color": colors, "line": {"color": CARD, "width": 1}},
        hovertext=hover,
        hoverinfo="text",
    ))

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": FONT_MONO, "size": 10},
        margin={"l": 10, "r": 20, "t": 16, "b": 40},
        xaxis={
            "title": {"text": "Delta to fastest (s)", "font": {"color": MUTED, "size": 11}},
            "gridcolor": "rgba(255,255,255,0.05)",
            "zerolinecolor": RED,
            "zerolinewidth": 1,
            "tickfont": {"color": MUTED, "size": 10},
        },
        yaxis={
            "tickfont": {"color": WHITE, "size": 10},
            "autorange": "reversed",
        },
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Race Map helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_track_outline(session):
    """
    Build a clean track outline using the session's fastest lap telemetry.
    Densifies to 2000 evenly-spaced parametric points so the outline is smooth.
    Returns (xs, ys) lists, or ([], []).
    """
    import numpy as np
    try:
        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry()
        xs = tel["X"].to_numpy(dtype=float)
        ys = tel["Y"].to_numpy(dtype=float)
        valid = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[valid], ys[valid]
        if len(xs) < 10:
            return [], []
        # Parametric densification → 2000 evenly-spaced points
        t_old = np.linspace(0, 1, len(xs))
        t_new = np.linspace(0, 1, 2000)
        return np.interp(t_new, t_old, xs).tolist(), np.interp(t_new, t_old, ys).tolist()
    except Exception:
        return [], []


def _build_map_frames(session):
    """
    Build per-frame car positions using continuous np.interp on a 10-second global timeline.

    Returns dict ready to store in dcc.Store:
    {
      "frames":    {"0": [{abbr, x, y, color, team, pos, lap}, ...], ...},
      "track_xs":  [...2000 pts...],
      "track_ys":  [...2000 pts...],
      "max_frame": N,
      "max_lap":   M,
      "step_s":    10,
    }
    """
    import numpy as np

    STEP_S = 1  # 1-second steps → dense enough for smooth animation

    laps = session.laps.copy()

    # ── Track outline ──────────────────────────────────────────────────────
    track_xs, track_ys = _get_track_outline(session)

    # ── Driver info ────────────────────────────────────────────────────────
    driver_info = {}
    for num in session.drivers:
        try:
            info = session.get_driver(str(num))
            driver_info[str(num)] = {
                "abbr":  info.get("Abbreviation", str(num)),
                "team":  info.get("TeamName", ""),
                "color": _TEAM_COLOR.get(info.get("TeamName", ""), MUTED2),
            }
        except Exception:
            driver_info[str(num)] = {"abbr": str(num), "team": "", "color": MUTED2}

    # ── Race time reference ────────────────────────────────────────────────
    # Use session start time (first LapStartDate) as t=0
    lap_start_dates = laps["LapStartDate"].dropna()
    if lap_start_dates.empty:
        return {}

    t0_raw = lap_start_dates.min()
    t0 = pd.Timestamp(t0_raw)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")

    # Estimate race end from last lap end (LapStartDate + LapTime)
    try:
        lap_ends = laps["LapStartDate"] + laps["LapTime"]
        t_end_raw = lap_ends.dropna().max()
        t_end = pd.Timestamp(t_end_raw)
        if t_end.tzinfo is None:
            t_end = t_end.tz_localize("UTC")
        else:
            t_end = t_end.tz_convert("UTC")
        duration_s = max(60.0, (t_end - t0).total_seconds())
    except Exception:
        duration_s = 7200.0  # fallback: 2 hours

    N = max(1, int(duration_s / STEP_S))
    t_global = np.linspace(0, duration_s, N + 1)

    # ── Per-driver interpolators ──────────────────────────────────────────
    interp_x = {}  # num → (t_arr, x_arr)
    interp_y = {}  # num → (t_arr, y_arr)

    for num in session.drivers:
        try:
            pos_df = session.laps.pick_drivers(num).get_pos_data()
            if pos_df is None or pos_df.empty:
                continue
            if isinstance(pos_df.index, pd.MultiIndex):
                pos_df = pos_df.reset_index()
            # Normalise Date column
            date_col = "Date" if "Date" in pos_df.columns else None
            if date_col is None:
                continue
            pos_df[date_col] = pd.to_datetime(pos_df[date_col], utc=True)
            pos_df["_t"] = (pos_df[date_col] - t0).dt.total_seconds()
            pos_df = (
                pos_df.dropna(subset=["_t", "X", "Y"])
                      .sort_values("_t")
                      .drop_duplicates("_t")
            )
            t_arr = pos_df["_t"].to_numpy(float)
            x_arr = pos_df["X"].to_numpy(float)
            y_arr = pos_df["Y"].to_numpy(float)
            if len(t_arr) < 2:
                continue
            interp_x[str(num)] = (t_arr, x_arr)
            interp_y[str(num)] = (t_arr, y_arr)
        except Exception:
            continue

    # ── Per-driver lap-number and race-position lookup ────────────────────
    # For each driver: sorted list of (t_seconds, lap_number) and (t_seconds, position)
    lap_lookup  = {}  # num → [(t, lap_num), ...]
    pos_lookup  = {}  # num → [(t, position), ...]
    max_lap_num = 1

    for num in session.drivers:
        drv_laps = laps[laps["DriverNumber"] == str(num)].sort_values("LapNumber")
        lap_entries = []
        pos_entries = []
        for _, r in drv_laps.iterrows():
            lap_t_raw = r.get("LapStartDate")
            lap_n = r.get("LapNumber")
            pos_n = r.get("Position")
            if pd.notna(lap_t_raw):
                ts = pd.Timestamp(lap_t_raw)
                if ts.tzinfo is None: ts = ts.tz_localize("UTC")
                else: ts = ts.tz_convert("UTC")
                t_sec = (ts - t0).total_seconds()
                if pd.notna(lap_n):
                    lap_entries.append((t_sec, int(lap_n)))
                    max_lap_num = max(max_lap_num, int(lap_n))
                if pd.notna(pos_n):
                    pos_entries.append((t_sec, int(pos_n)))
        lap_lookup[str(num)] = lap_entries
        pos_lookup[str(num)] = pos_entries

    def _step_at(entries, t):
        """Step-function lookup: return value at time t from sorted (t, val) list."""
        if not entries:
            return None
        val = entries[0][1]
        for et, ev in entries:
            if et <= t:
                val = ev
            else:
                break
        return val

    # ── Build frames ───────────────────────────────────────────────────────
    # Fully vectorised: interpolate every driver's entire timeline at once,
    # then build lap/pos arrays using numpy searchsorted (no Python loops per frame).
    all_drivers = []
    for num in session.drivers:
        key = str(num)
        if key not in interp_x:
            continue
        t_arr, x_arr = interp_x[key]
        _,     y_arr = interp_y[key]
        info = driver_info.get(key, {"abbr": key, "team": "", "color": MUTED2})

        # Vectorised X/Y across all time steps
        xs_all = np.interp(t_global, t_arr, x_arr)
        ys_all = np.interp(t_global, t_arr, y_arr)
        # NaN outside driver's telemetry window
        outside = (t_global < t_arr[0] - STEP_S) | (t_global > t_arr[-1] + STEP_S)
        xs_all[outside] = np.nan
        ys_all[outside] = np.nan

        # Vectorised lap & pos via searchsorted
        def _vectorise_lookup(entries):
            if not entries:
                return np.full(len(t_global), None, dtype=object)
            t_e = np.array([e[0] for e in entries], dtype=float)
            v_e = np.array([e[1] for e in entries])
            # searchsorted gives index where t_global would be inserted;
            # clip to last valid index for step-function semantics
            idx = np.searchsorted(t_e, t_global, side="right") - 1
            idx = np.clip(idx, 0, len(v_e) - 1)
            result = v_e[idx]
            # NaN before the first entry
            result = result.astype(object)
            result[t_global < t_e[0]] = None
            return result

        laps_arr = _vectorise_lookup(lap_lookup.get(key, []))
        pos_arr  = _vectorise_lookup(pos_lookup.get(key, []))

        all_drivers.append({
            "abbr":  info["abbr"],
            "team":  info["team"],
            "color": info["color"],
            "xs":    xs_all,
            "ys":    ys_all,
            "laps":  laps_arr,
            "pos":   pos_arr,
        })

    frames = {}
    for i in range(len(t_global)):
        cars = []
        for d in all_drivers:
            x = d["xs"][i]
            y = d["ys"][i]
            if np.isnan(x) or np.isnan(y):
                continue
            lap = d["laps"][i]
            pos = d["pos"][i]
            cars.append({
                "abbr":  d["abbr"],
                "team":  d["team"],
                "color": d["color"],
                "x":     round(float(x), 1),
                "y":     round(float(y), 1),
                "pos":   int(pos) if pos is not None else "?",
                "lap":   int(lap) if lap is not None else "?",
            })
        frames[str(i)] = cars

    # ── Slider marks (label by lap, ~20 marks total) ───────────────────────
    marks = {}
    first_driver = str(session.drivers[0]) if session.drivers else None
    mark_step = max(1, N // 20)
    for m in range(0, N + 1, mark_step):
        t = t_global[min(m, len(t_global) - 1)]
        lap = _step_at(lap_lookup.get(first_driver, []), t) if first_driver else None
        marks[m] = {
            "label": f"L{lap}" if lap else "",
            "style": {"color": MUTED, "fontSize": "9px"},
        }
    marks[0] = {"label": "L1", "style": {"color": MUTED, "fontSize": "9px"}}
    marks[N] = {"label": f"L{max_lap_num}", "style": {"color": MUTED, "fontSize": "9px"}}

    return {
        "frames":    frames,
        "track_xs":  track_xs,
        "track_ys":  track_ys,
        "max_frame": N,
        "max_lap":   max_lap_num,
        "step_s":    STEP_S,
        "marks":     marks,
    }


def _build_race_map_figure(map_data_for_lap, track_xs, track_ys, lap_num, max_lap):
    """Build a Plotly figure showing car positions on the track for a given lap."""
    fig = go.Figure()

    # Track outline
    if track_xs and track_ys:
        fig.add_trace(go.Scatter(
            x=track_xs,
            y=track_ys,
            mode="lines",
            line={"color": "rgba(255,255,255,0.12)", "width": 2},
            hoverinfo="skip",
            showlegend=False,
            name="track",
        ))

    # Car dots
    if map_data_for_lap:
        # Sort by position so P1 is drawn last (on top)
        cars = sorted(map_data_for_lap, key=lambda c: c["pos"] if isinstance(c["pos"], int) else 99)

        for car in cars:
            pos_label = f"P{car['pos']}" if isinstance(car["pos"], int) else "?"
            hover = f"<b>{pos_label} {car['abbr']}</b><br>{car['team']}<extra></extra>"
            fig.add_trace(go.Scatter(
                x=[car["x"]],
                y=[car["y"]],
                mode="markers+text",
                marker={
                    "color": car["color"],
                    "size": 10,
                    "line": {"color": "#000", "width": 1.5},
                    "symbol": "circle",
                },
                text=[car["abbr"]],
                textposition="top center",
                textfont={"color": "#fff", "size": 8, "family": "JetBrains Mono"},
                name=car["abbr"],
                hovertemplate=hover,
                showlegend=False,
            ))

    # Fixed axis ranges based on track outline
    if track_xs and track_ys:
        pad_x = (max(track_xs) - min(track_xs)) * 0.05
        pad_y = (max(track_ys) - min(track_ys)) * 0.05
        x_range = [min(track_xs) - pad_x, max(track_xs) + pad_x]
        y_range = [min(track_ys) - pad_y, max(track_ys) + pad_y]
    else:
        x_range = None
        y_range = None

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": FONT_MONO, "size": 10},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "showgrid": False, "zeroline": False, "showticklabels": False,
            "scaleanchor": "y", "scaleratio": 1,
            **({"range": x_range} if x_range else {}),
        },
        yaxis={
            "showgrid": False, "zeroline": False, "showticklabels": False,
            **({"range": y_range} if y_range else {}),
        },
        showlegend=False,
        hovermode="closest",
        uirevision="track",  # keep zoom/pan across updates
    )
    return fig


def _build_animated_figure(payload):
    """
    Build a fully self-contained animated go.Figure.
    All frames are embedded — playback is 100% client-side (no Python round-trips).

    Uses Plotly's native:
      - fig.frames  → one frame per time-step
      - layout.updatemenus  → Play / Pause buttons
      - layout.sliders  → scrub bar with lap labels
    """
    import numpy as np

    frames_data = payload.get("frames", {})
    track_xs    = payload.get("track_xs", [])
    track_ys    = payload.get("track_ys", [])
    max_frame   = payload.get("max_frame", 1)
    max_lap     = payload.get("max_lap", "?")
    step_s      = payload.get("step_s", 10)

    # ── Axis ranges from track outline ────────────────────────────────────
    if track_xs and track_ys:
        pad_x = (max(track_xs) - min(track_xs)) * 0.06
        pad_y = (max(track_ys) - min(track_ys)) * 0.06
        x_range = [min(track_xs) - pad_x, max(track_xs) + pad_x]
        y_range = [min(track_ys) - pad_y, max(track_ys) + pad_y]
    else:
        x_range = None
        y_range = None

    # ── Stable driver ordering (from frame 0) ─────────────────────────────
    # Sort by position so trace order is consistent across all frames
    frame0_cars = sorted(
        frames_data.get("0", []),
        key=lambda c: c["pos"] if isinstance(c["pos"], int) else 99,
    )
    driver_abbrs = [c["abbr"] for c in frame0_cars]

    def _car_traces(cars_list):
        """Build one go.Scatter per car from a list of car dicts."""
        # Index by abbr for stable ordering
        by_abbr = {c["abbr"]: c for c in cars_list}
        traces = []
        for abbr in driver_abbrs:
            car = by_abbr.get(abbr)
            if car is None:
                # Driver retired / no data — place off-screen
                traces.append(go.Scatter(
                    x=[None], y=[None],
                    mode="markers+text",
                    marker={"color": MUTED2, "size": 10, "line": {"color": "#000", "width": 1}},
                    text=[abbr],
                    textposition="top center",
                    textfont={"color": MUTED2, "size": 8, "family": "JetBrains Mono"},
                    hoverinfo="skip",
                    showlegend=False,
                ))
            else:
                pos_label = f"P{car['pos']}" if isinstance(car["pos"], int) else "?"
                lap_label = f"Lap {car['lap']}" if isinstance(car.get("lap"), int) else ""
                traces.append(go.Scatter(
                    x=[car["x"]],
                    y=[car["y"]],
                    mode="markers+text",
                    marker={
                        "color": car["color"],
                        "size": 11,
                        "line": {"color": "rgba(0,0,0,0.8)", "width": 1.5},
                        "symbol": "circle",
                    },
                    text=[abbr],
                    textposition="top center",
                    textfont={"color": "#ffffff", "size": 8, "family": "JetBrains Mono"},
                    hovertemplate=f"<b>{pos_label} {abbr}</b><br>{car['team']}<br>{lap_label}<extra></extra>",
                    showlegend=False,
                ))
        return traces

    # ── Track outline trace (trace index 0, never animated) ───────────────
    track_trace = go.Scatter(
        x=track_xs,
        y=track_ys,
        mode="lines",
        line={"color": "rgba(255,255,255,0.15)", "width": 2.5},
        hoverinfo="skip",
        showlegend=False,
        name="_track",
    )

    # ── Initial frame data ─────────────────────────────────────────────────
    initial_car_traces = _car_traces(frame0_cars)
    initial_data = [track_trace] + initial_car_traces

    # ── Plotly frames (one per time step) ─────────────────────────────────
    # Each frame only updates the car traces (indices 1..N_drivers), not trace 0
    n_drivers = len(driver_abbrs)
    trace_indices = list(range(1, n_drivers + 1))

    plotly_frames = []
    # Build a "lap N / M" label for each frame
    for i in range(max_frame + 1):
        cars_list = frames_data.get(str(i), [])
        car_traces = _car_traces(cars_list)
        # Derive lap from first car with a known lap
        cur_lap = "?"
        for c in cars_list:
            if isinstance(c.get("lap"), int):
                cur_lap = c["lap"]
                break
        frame_name = str(i)
        frame_label = f"Lap {cur_lap} / {max_lap}"
        plotly_frames.append(go.Frame(
            data=car_traces,
            traces=trace_indices,
            name=frame_name,
            layout=go.Layout(
                annotations=[{
                    "text": frame_label,
                    "x": 0.01, "y": 0.99,
                    "xref": "paper", "yref": "paper",
                    "xanchor": "left", "yanchor": "top",
                    "showarrow": False,
                    "font": {"color": WHITE, "size": 13, "family": FONT_MONO},
                    "bgcolor": "rgba(0,0,0,0)",
                }]
            ),
        ))

    # ── Slider steps ───────────────────────────────────────────────────────
    # One step per frame; label only shown at lap boundaries
    slider_steps = []
    prev_lap = None
    for i in range(max_frame + 1):
        cars_list = frames_data.get(str(i), [])
        cur_lap = None
        for c in cars_list:
            if isinstance(c.get("lap"), int):
                cur_lap = c["lap"]
                break
        show_label = cur_lap is not None and cur_lap != prev_lap
        slider_steps.append({
            "args": [
                [str(i)],
                {"frame": {"duration": 0, "redraw": False},
                 "mode": "immediate",
                 "transition": {"duration": 0}},
            ],
            "label": f"L{cur_lap}" if show_label else "",
            "method": "animate",
        })
        if show_label:
            prev_lap = cur_lap

    # ── Assemble figure ────────────────────────────────────────────────────
    fig = go.Figure(data=initial_data, frames=plotly_frames)

    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": FONT_MONO, "size": 10},
        margin={"l": 8, "r": 8, "t": 8, "b": 60},
        xaxis={
            "showgrid": False, "zeroline": False, "showticklabels": False,
            "scaleanchor": "y", "scaleratio": 1,
            **({"range": x_range} if x_range else {}),
        },
        yaxis={
            "showgrid": False, "zeroline": False, "showticklabels": False,
            **({"range": y_range} if y_range else {}),
        },
        showlegend=False,
        hovermode="closest",
        # ── Play / Pause buttons ────────────────────────────────────────
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": -0.06,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"r": 10, "t": 0},
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 40, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        },
                    ],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
            "bgcolor": RED,
            "bordercolor": "rgba(0,0,0,0)",
            "font": {
                "color": WHITE,
                "size": 11,
                "family": FONT_SANS,
            },
        }],
        # ── Scrub slider ────────────────────────────────────────────────
        sliders=[{
            "active": 0,
            "currentvalue": {"visible": False},
            "pad": {"b": 4, "t": 4, "l": 110},
            "x": 0.0, "y": 0.0,
            "len": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "steps": slider_steps,
            "bgcolor": CARD2,
            "bordercolor": BORDER2,
            "tickcolor": MUTED,
            "font": {"color": MUTED, "size": 9, "family": FONT_MONO},
        }],
    )

    return fig


def build_race_map_section():
    """Returns the race map card. The animated figure with play/slider is loaded by the load_race callback."""
    return html.Div(
        id="race-map-section",
        style={"display": "none"},
        children=[
            _card(children=[
                _section_header("Race Map", badge="real-time car positions · all drivers · client-side animation"),
                dcc.Graph(
                    id="race-map-graph",
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines", "autoScale2d"],
                        "displaylogo": False,
                        "scrollZoom": True,
                    },
                    style={"height": "560px"},
                    figure=_empty_figure(""),
                ),
            ]),
        ],
    )


def _build_incidents_panel(rc_all):
    """Timeline of race control events (flags, SC, VSC, penalties)."""
    if not rc_all:
        return html.Div()

    # Filter interesting events
    interesting = []
    skip_msgs = {"drs enabled", "drs disabled", "pit exit open", "pit exit closed"}
    for item in rc_all:
        msg = str(item.get("message", "")).strip()
        if not msg or msg.lower() in skip_msgs:
            continue
        try:
            dt = pd.to_datetime(item.get("date", ""))
        except Exception:
            dt = None
        interesting.append({
            "time": dt,
            "message": msg,
            "category": str(item.get("category", "")),
            "flag": str(item.get("flag", "")),
        })

    if not interesting:
        return html.Div()

    def _flag_color(flag, category, message):
        f = flag.lower()
        m = message.lower()
        c = category.lower()
        if "red" in f or "red" in m:           return "#FF3333"
        if "yellow" in f or "yellow" in m:     return YELLOW
        if "safety car" in m or "sc" in c:     return YELLOW
        if "vsc" in m or "virtual" in m:       return ORANGE
        if "chequered" in f or "chequered" in m: return WHITE
        if "penalty" in m or "time penalty" in m: return ORANGE
        if "investigation" in m:               return TEAL
        if "green" in f:                       return GREEN
        return MUTED

    rows = []
    for ev in interesting:
        time_str = ev["time"].strftime("%H:%M:%S") if ev["time"] and not pd.isna(ev["time"]) else "—"
        color = _flag_color(ev["flag"], ev["category"], ev["message"])
        rows.append(html.Div(
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "gap": "14px",
                "padding": "9px 0",
                "borderBottom": f"1px solid {BORDER}",
            },
            children=[
                html.Div(time_str, style={
                    "color": MUTED,
                    "fontSize": "10px",
                    "fontFamily": FONT_MONO,
                    "minWidth": "64px",
                    "paddingTop": "1px",
                    "letterSpacing": "0.06em",
                }),
                html.Div(style={
                    "width": "3px",
                    "minHeight": "16px",
                    "backgroundColor": color,
                    "borderRadius": "2px",
                    "flexShrink": "0",
                    "marginTop": "2px",
                }),
                html.Div(ev["message"], style={
                    "color": WHITE,
                    "fontSize": "11px",
                    "fontFamily": FONT_SANS,
                    "lineHeight": "1.5",
                    "flex": "1",
                }),
            ],
        ))

    return _card(children=[
        _section_header("Race Incidents", badge="race control · flags · penalties"),
        html.Div(
            style={"maxHeight": "320px", "overflowY": "auto"},
            children=rows,
        ),
    ])
