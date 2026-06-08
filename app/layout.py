from dash import dcc, html
import dash_bootstrap_components as dbc

# ── Colour palette ──────────────────────────────────────────────────────────
BG      = "#0A0A0A"      # True OLED black
SURFACE = "#111111"      # Header / nav surface
CARD    = "#161616"      # Card / panel background
CARD2   = "#1A1A1A"      # Slightly elevated card (inner sections)
BORDER  = "#252525"      # Subtle border
BORDER2 = "#333333"      # More visible divider
RED     = "#E10600"      # F1 primary red
RED_DIM = "#7A0300"      # Dimmed red
WHITE   = "#F0F0F0"      # Primary text
GREY    = "#707070"      # Secondary / label text
GREY2   = "#4A4A4A"      # Muted / disabled text
GREEN   = "#00D264"      # Positive / fastest lap
ORANGE  = "#FF8C00"      # Medium stress / warning
YELLOW  = "#FFD700"  # P1 / gold

# Tyre compound colours (kept for callbacks)
TYRE_SOFT   = "#E10600"
TYRE_MEDIUM = "#FFF200"
TYRE_HARD   = "#F0F0F0"

# ── Typography ───────────────────────────────────────────────────────────────
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace"
FONT_SANS = "'Fira Sans', 'Inter', system-ui, sans-serif"

# ── Shared style helpers ─────────────────────────────────────────────────────
LABEL_STYLE = {
    "color": GREY,
    "fontSize": "10px",
    "letterSpacing": "0.18em",
    "textTransform": "uppercase",
    "marginBottom": "6px",
    "fontFamily": FONT_MONO,
    "fontWeight": "500",
}

SECTION_TITLE_STYLE = {
    **LABEL_STYLE,
    "fontSize": "9px",
    "letterSpacing": "0.22em",
    "borderLeft": f"2px solid {RED}",
    "paddingLeft": "8px",
    "marginBottom": "0",
    "lineHeight": "1",
}

DROPDOWN_STYLE = {
    "backgroundColor": CARD2,
    "color": RED,
    "border": f"1px solid {BORDER2}",
    "borderRadius": "4px",
    "fontFamily": FONT_MONO,
    "fontSize": "12px",
}

INPUT_STYLE = {
    **DROPDOWN_STYLE,
    "padding": "8px 12px",
    "outline": "none",
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
}


def _dropdown(id_, placeholder, options=None, value=None, clearable=False):
    return dcc.Dropdown(
        id=id_,
        options=options or [],
        value=value,
        placeholder=placeholder,
        clearable=clearable,
        style=DROPDOWN_STYLE,
        className="f1-dropdown",
    )


def _stat_chip(chip_id: str, label: str, icon: str = ""):
    """Metric chip: label on top, large value below, optional accent icon."""
    return html.Div(
        id=chip_id,
        style={
            "backgroundColor": CARD,
            "border": f"1px solid {BORDER}",
            "borderTop": f"2px solid {BORDER2}",
            "borderRadius": "6px",
            "padding": "10px 18px",
            "minWidth": "110px",
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "gap": "4px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                children=[
                    html.Span(icon, style={"fontSize": "10px", "opacity": "0.6"}) if icon else None,
                    html.Span(label, style={**LABEL_STYLE, "marginBottom": "0"}),
                ],
            ),
            html.Div("—", id=f"{chip_id}-val", style={
                "color": WHITE,
                "fontSize": "16px",
                "fontWeight": "700",
                "fontFamily": FONT_MONO,
                "letterSpacing": "0.04em",
                "lineHeight": "1.2",
            }),
        ],
    )


def _panel_header(title: str, subtitle: str = ""):
    """Section panel header with red left accent and optional subtitle."""
    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "marginBottom": "14px",
            "paddingBottom": "10px",
            "borderBottom": f"1px solid {BORDER}",
        },
        children=[
            html.Div(title, style=SECTION_TITLE_STYLE),
            html.Div(subtitle, style={
                "color": GREY2,
                "fontSize": "9px",
                "letterSpacing": "0.1em",
                "fontFamily": FONT_MONO,
                "textTransform": "uppercase",
            }) if subtitle else None,
        ],
    )


def _card(children, flex=None, min_width=None, extra_style=None):
    style = {
        "backgroundColor": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": "16px",
    }
    if flex:
        style["flex"] = flex
    if min_width:
        style["minWidth"] = min_width
    if extra_style:
        style.update(extra_style)
    return html.Div(style=style, children=children)


def build_layout():
    return html.Div(
        style={
            "backgroundColor": BG,
            "minHeight": "100vh",
            "fontFamily": FONT_MONO,
            "color": WHITE,
        },
        children=[

            # ── Header ──────────────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"2px solid {RED}",
                    "padding": "0 28px",
                    "height": "52px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "position": "sticky",
                    "top": "0",
                    "zIndex": "100",
                },
                children=[
                    # Left: logo mark + title
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "14px"},
                        children=[
                            # F1 flag-stripe logo mark
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "3px"},
                                children=[
                                    html.Div(style={"width": "20px", "height": "3px", "backgroundColor": RED}),
                                    html.Div(style={"width": "20px", "height": "3px", "backgroundColor": WHITE}),
                                    html.Div(style={"width": "20px", "height": "3px", "backgroundColor": RED}),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "1px"},
                                children=[
                                    html.Span("F1 STRESS ANALYZER", style={
                                        "color": WHITE,
                                        "fontSize": "13px",
                                        "fontWeight": "700",
                                        "letterSpacing": "0.22em",
                                        "lineHeight": "1",
                                    }),
                                    html.Span("AI · TELEMETRY · RADIO ANALYSIS", style={
                                        "color": GREY,
                                        "fontSize": "8px",
                                        "letterSpacing": "0.2em",
                                        "lineHeight": "1",
                                    }),
                                ],
                            ),
                        ],
                    ),
                    # Right: session status badge
                    html.Div(
                        id="status-msg",
                        style={
                            "color": GREY,
                            "fontSize": "10px",
                            "letterSpacing": "0.12em",
                            "fontFamily": FONT_MONO,
                            "maxWidth": "400px",
                            "textAlign": "right",
                        },
                    ),
                ],
            ),

            # ── Controls bar ─────────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"1px solid {BORDER}",
                    "padding": "12px 28px",
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "flex-end",
                    "flexWrap": "wrap",
                },
                children=[
                    # Season
                    html.Div([
                        html.Div("Season", style=LABEL_STYLE),
                        _dropdown(
                            "dd-year", "Year",
                            options=[{"label": str(y), "value": y} for y in [2023, 2024, 2025]],
                            value=2024,
                        ),
                    ], style={"minWidth": "110px", "width": "110px"}),

                    # Grand Prix
                    html.Div([
                        html.Div("Grand Prix", style=LABEL_STYLE),
                        _dropdown("dd-race", "Select race…"),
                    ], style={"minWidth": "220px", "flex": "2"}),

                    # Vertical divider
                    html.Div(style={
                        "width": "1px",
                        "height": "36px",
                        "backgroundColor": BORDER2,
                        "alignSelf": "flex-end",
                        "marginBottom": "2px",
                    }),

                    # Driver 1
                    html.Div([
                        html.Div("Driver", style=LABEL_STYLE),
                        dcc.Input(
                            id="input-driver",
                            type="text",
                            value="LEC",
                            maxLength=3,
                            placeholder="LEC",
                            style={**INPUT_STYLE, "width": "72px"},
                        ),
                    ]),

                    # VS label
                    html.Div("VS", style={
                        "color": GREY2,
                        "fontSize": "9px",
                        "letterSpacing": "0.2em",
                        "alignSelf": "flex-end",
                        "paddingBottom": "10px",
                        "fontFamily": FONT_MONO,
                    }),

                    # Driver 2
                    html.Div([
                        html.Div("Compare", style=LABEL_STYLE),
                        dcc.Input(
                            id="input-driver2",
                            type="text",
                            value="",
                            maxLength=3,
                            placeholder="optional",
                            style={**INPUT_STYLE, "width": "96px"},
                        ),
                    ]),

                    # Vertical divider
                    html.Div(style={
                        "width": "1px",
                        "height": "36px",
                        "backgroundColor": BORDER2,
                        "alignSelf": "flex-end",
                        "marginBottom": "2px",
                    }),

                    # Analyze button
                    html.Div([
                        html.Div("\u00a0", style={**LABEL_STYLE, "visibility": "hidden"}),
                        html.Button(
                            children=[
                                html.Span("ANALYZE", style={"letterSpacing": "0.18em"}),
                            ],
                            id="btn-analyze",
                            n_clicks=0,
                            style={
                                "backgroundColor": RED,
                                "color": WHITE,
                                "border": "none",
                                "padding": "8px 24px",
                                "fontFamily": FONT_MONO,
                                "fontSize": "11px",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "borderRadius": "4px",
                                "transition": "background 0.15s, transform 0.1s",
                                "outline": "none",
                                "height": "36px",
                            },
                        ),
                    ]),
                ],
            ),

            # ── Stats bar (hidden until analysis runs) ────────────────────────
            html.Div(
                id="stats-bar",
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"1px solid {BORDER}",
                    "padding": "10px 28px",
                    "display": "none",
                    "gap": "10px",
                    "flexWrap": "wrap",
                    "alignItems": "stretch",
                },
                children=[
                    _stat_chip("chip-peak-stress", "Peak Stress"),
                    _stat_chip("chip-avg-stress",  "Avg Stress"),
                    _stat_chip("chip-fastest-lap", "Fastest Lap"),
                    _stat_chip("chip-pit-stops",   "Pit Stops"),
                    _stat_chip("chip-radio-count", "Radio Msgs"),
                ],
            ),

            # ── Main charts row ───────────────────────────────────────────────
            html.Div(
                style={
                    "padding": "20px 28px 0 28px",
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                },
                children=[

                    # Telemetry & AI stress (wide)
                    _card(
                        flex="6",
                        min_width="460px",
                        children=[
                            _panel_header(
                                "Telemetry & AI Stress",
                                "speed · throttle · brake · gear · drs",
                            ),
                            dcc.Graph(
                                id="chart-telemetry",
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": [
                                        "select2d", "lasso2d", "autoScale2d",
                                        "hoverClosestCartesian", "hoverCompareCartesian",
                                        "toggleSpikelines",
                                    ],
                                    "displaylogo": False,
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": "f1_telemetry",
                                    },
                                },
                                style={"height": "500px"},
                                figure=_empty_figure(
                                    "Select a season, race and driver\nthen click ANALYZE"
                                ),
                            ),
                        ],
                    ),

                    # Track map (narrower)
                    _card(
                        flex="4",
                        min_width="300px",
                        children=[
                            _panel_header(
                                "Track Map",
                                "click a marker to load radio",
                            ),
                            dcc.Graph(
                                id="chart-track",
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": [
                                        "select2d", "lasso2d", "toggleSpikelines",
                                    ],
                                    "displaylogo": False,
                                },
                                style={"height": "500px"},
                                figure=_empty_figure(""),
                            ),
                        ],
                    ),
                ],
            ),

            # ── Radio panel (click-to-expand) ─────────────────────────────────
            html.Div(
                id="radio-panel",
                style={"padding": "16px 28px 0 28px"},
                children=[],
            ),

            # ── Lap analysis row ───────────────────────────────────────────────
            html.Div(
                style={
                    "padding": "16px 28px 0 28px",
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                },
                children=[

                    # Lap-by-lap stress
                    _card(
                        flex="1",
                        min_width="280px",
                        children=[
                            _panel_header("Lap-by-Lap Stress", "avg ai stress per lap"),
                            dcc.Graph(
                                id="chart-lap-stress",
                                config={"displayModeBar": False, "displaylogo": False},
                                style={"height": "180px"},
                                figure=_empty_figure(""),
                            ),
                        ],
                    ),

                    # Lap time evolution
                    _card(
                        flex="1",
                        min_width="280px",
                        children=[
                            _panel_header("Lap Time Evolution", "seconds per lap"),
                            dcc.Graph(
                                id="chart-lap-times",
                                config={"displayModeBar": False, "displaylogo": False},
                                style={"height": "180px"},
                                figure=_empty_figure(""),
                            ),
                        ],
                    ),
                ],
            ),

            # ── Leaderboard panel ─────────────────────────────────────────────
            html.Div(
                id="leaderboard-panel",
                style={"padding": "16px 28px 36px 28px"},
                children=[],
            ),

            # ── Hidden data stores ────────────────────────────────────────────
            dcc.Store(id="store-radio"),
            dcc.Store(id="store-session-meta"),
            dcc.Store(id="store-lap-stress"),
            dcc.Store(id="store-incidents"),
            dcc.Store(id="store-leaderboard"),
            dcc.Store(id="store-lap-times"),
        ],
    )


def _empty_figure(annotation: str):
    """Return a blank dark figure with an optional centred annotation."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": FONT_MONO},
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        xaxis={
            "showgrid": False, "zeroline": False,
            "showticklabels": False, "color": GREY,
        },
        yaxis={
            "showgrid": False, "zeroline": False,
            "showticklabels": False, "color": GREY,
        },
    )
    if annotation:
        fig.add_annotation(
            text=annotation.replace("\n", "<br>"),
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font={"color": GREY2, "size": 12, "family": FONT_MONO},
            align="center",
        )
    return fig
