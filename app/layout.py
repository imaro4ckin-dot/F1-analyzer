from dash import dcc, html
import dash_bootstrap_components as dbc

# ── Colour palette ──────────────────────────────────────────────────────────
BG = "#0D0D0D"
SURFACE = "#161616"
CARD = "#1C1C1C"
BORDER = "#2A2A2A"
RED = "#E10600"
RED_DIM = "#7A0300"
WHITE = "#F5F5F5"
GREY = "#888888"
GREEN = "#39FF14"
ORANGE = "#FF6B35"
YELLOW = "#FFD700"
TYRE_SOFT = "#E10600"
TYRE_MEDIUM = "#FFF200"
TYRE_HARD = "#F5F5F5"

# ── Shared style helpers ─────────────────────────────────────────────────────
LABEL_STYLE = {
    "color": GREY,
    "fontSize": "10px",
    "letterSpacing": "0.15em",
    "textTransform": "uppercase",
    "marginBottom": "4px",
    "fontFamily": "'JetBrains Mono', 'Courier New', monospace",
}

DROPDOWN_STYLE = {
    "backgroundColor": CARD,
    "color": RED,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
    "fontFamily": "'JetBrains Mono', 'Courier New', monospace",
    "fontSize": "13px",
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


def _stat_chip(chip_id: str, label: str):
    """Return a stats bar metric chip with a label and updatable value."""
    return html.Div(
        id=chip_id,
        style={
            "backgroundColor": CARD,
            "border": f"1px solid {BORDER}",
            "borderRadius": "6px",
            "padding": "8px 16px",
            "minWidth": "120px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "2px",
        },
        children=[
            html.Div(label, style={**LABEL_STYLE, "marginBottom": "0"}),
            html.Div("—", id=f"{chip_id}-val", style={
                "color": WHITE,
                "fontSize": "14px",
                "fontWeight": "700",
                "fontFamily": "'JetBrains Mono', 'Courier New', monospace",
                "letterSpacing": "0.05em",
            }),
        ],
    )


def build_layout():
    return html.Div(
        style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "'JetBrains Mono', 'Courier New', monospace"},
        children=[
            # ── Header ──────────────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"2px solid {RED}",
                    "padding": "0 32px",
                    "height": "60px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                },
                children=[
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
                        html.Div(style={"width": "4px", "height": "32px", "backgroundColor": RED}),
                        html.Span("F1 STRESS ANALYZER", style={
                            "color": WHITE, "fontSize": "15px", "fontWeight": "700",
                            "letterSpacing": "0.2em",
                        }),
                    ]),
                    html.Span("AI · TELEMETRY · RADIO", style={
                        "color": GREY, "fontSize": "10px", "letterSpacing": "0.25em",
                    }),
                ],
            ),

            # ── Controls bar ────────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"1px solid {BORDER}",
                    "padding": "16px 32px",
                    "display": "flex",
                    "gap": "16px",
                    "alignItems": "flex-end",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div([
                        html.Div("SEASON", style=LABEL_STYLE),
                        _dropdown(
                            "dd-year", "Year",
                            options=[{"label": str(y), "value": y} for y in [2023, 2024, 2025]],
                            value=2024,
                        ),
                    ], style={"minWidth": "120px"}),

                    html.Div([
                        html.Div("GRAND PRIX", style=LABEL_STYLE),
                        _dropdown("dd-race", "Select race…"),
                    ], style={"minWidth": "240px", "flex": "1"}),

                    html.Div([
                        html.Div("DRIVER", style=LABEL_STYLE),
                        dcc.Input(
                            id="input-driver",
                            type="text",
                            value="LEC",
                            maxLength=3,
                            placeholder="VER",
                            style={
                                **DROPDOWN_STYLE,
                                "padding": "8px 12px",
                                "width": "80px",
                                "textTransform": "uppercase",
                            },
                        ),
                    ]),

                    html.Div([
                        html.Div("VS DRIVER", style=LABEL_STYLE),
                        dcc.Input(
                            id="input-driver2",
                            type="text",
                            value="",
                            maxLength=3,
                            placeholder="VER (opt.)",
                            style={
                                **DROPDOWN_STYLE,
                                "padding": "8px 12px",
                                "width": "100px",
                                "textTransform": "uppercase",
                            },
                        ),
                    ]),

                    html.Div([
                        html.Div("\u00a0", style=LABEL_STYLE),
                        html.Button(
                            "ANALYZE",
                            id="btn-analyze",
                            n_clicks=0,
                            style={
                                "backgroundColor": RED,
                                "color": WHITE,
                                "border": "none",
                                "padding": "9px 28px",
                                "fontFamily": "'JetBrains Mono', 'Courier New', monospace",
                                "fontSize": "12px",
                                "fontWeight": "700",
                                "letterSpacing": "0.15em",
                                "cursor": "pointer",
                                "borderRadius": "4px",
                                "transition": "background 0.15s",
                            },
                        ),
                    ]),

                    # Status message
                    html.Div(id="status-msg", style={
                        "color": GREY, "fontSize": "11px", "letterSpacing": "0.1em",
                        "alignSelf": "center", "marginLeft": "8px",
                    }),
                ],
            ),

            # ── Stats bar (hidden until analysis runs) ──────────────────────
            html.Div(
                id="stats-bar",
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"1px solid {BORDER}",
                    "padding": "10px 32px",
                    "display": "none",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                },
                children=[
                    _stat_chip("chip-peak-stress",  "PEAK STRESS"),
                    _stat_chip("chip-avg-stress",   "AVG STRESS"),
                    _stat_chip("chip-fastest-lap",  "FASTEST LAP"),
                    _stat_chip("chip-pit-stops",    "PIT STOPS"),
                    _stat_chip("chip-radio-count",  "RADIO MSGS"),
                ],
            ),

            # ── Main content ────────────────────────────────────────────────
            html.Div(
                style={"padding": "20px 32px", "display": "flex", "gap": "20px", "flexWrap": "wrap"},
                children=[

                    # Left: Telemetry chart
                    html.Div(
                        style={
                            "flex": "6",
                            "minWidth": "460px",
                            "backgroundColor": CARD,
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "6px",
                            "padding": "16px",
                        },
                        children=[
                            html.Div("TELEMETRY & AI STRESS", style={**LABEL_STYLE, "marginBottom": "12px"}),
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
                                    "toImageButtonOptions": {"format": "png", "filename": "f1_telemetry"},
                                },
                                style={"height": "520px"},
                                figure=_empty_figure("Select a race and driver, then click ANALYZE"),
                            ),
                        ],
                    ),

                    # Right: Track map
                    html.Div(
                        style={
                            "flex": "4",
                            "minWidth": "320px",
                            "backgroundColor": CARD,
                            "border": f"1px solid {BORDER}",
                            "borderRadius": "6px",
                            "padding": "16px",
                        },
                        children=[
                            html.Div("TRACK MAP  ·  RADIO EVENTS", style={**LABEL_STYLE, "marginBottom": "12px"}),
                            dcc.Graph(
                                id="chart-track",
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"],
                                    "displaylogo": False,
                                },
                                style={"height": "520px"},
                                figure=_empty_figure(""),
                            ),
                        ],
                    ),
                ],
            ),

            # ── Radio panel ─────────────────────────────────────────────────
            html.Div(
                id="radio-panel",
                style={"padding": "0 32px 20px 32px"},
                children=[],
            ),

            # ── Lap charts: stress + lap time side-by-side ──────────────────
            html.Div(
                style={"padding": "0 32px 20px 32px"},
                children=[
                    html.Div(
                        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                        children=[
                            # Left: Lap-by-lap stress
                            html.Div(
                                style={
                                    "flex": "1",
                                    "minWidth": "300px",
                                    "backgroundColor": CARD,
                                    "border": f"1px solid {BORDER}",
                                    "borderRadius": "6px",
                                    "padding": "16px",
                                },
                                children=[
                                    html.Div("LAP-BY-LAP STRESS", style={**LABEL_STYLE, "marginBottom": "8px"}),
                                    dcc.Graph(
                                        id="chart-lap-stress",
                                        config={"displayModeBar": False, "displaylogo": False},
                                        style={"height": "160px"},
                                        figure=_empty_figure(""),
                                    ),
                                ],
                            ),
                            # Right: Lap time evolution
                            html.Div(
                                style={
                                    "flex": "1",
                                    "minWidth": "300px",
                                    "backgroundColor": CARD,
                                    "border": f"1px solid {BORDER}",
                                    "borderRadius": "6px",
                                    "padding": "16px",
                                },
                                children=[
                                    html.Div("LAP TIME EVOLUTION", style={**LABEL_STYLE, "marginBottom": "8px"}),
                                    dcc.Graph(
                                        id="chart-lap-times",
                                        config={"displayModeBar": False, "displaylogo": False},
                                        style={"height": "160px"},
                                        figure=_empty_figure(""),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # ── Leaderboard panel ────────────────────────────────────────────
            html.Div(
                id="leaderboard-panel",
                style={"padding": "0 32px 32px 32px"},
                children=[],
            ),

            # ── Hidden data stores ───────────────────────────────────────────
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
        font={"color": WHITE, "family": "JetBrains Mono, Courier New, monospace"},
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "color": GREY},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "color": GREY},
    )
    if annotation:
        fig.add_annotation(
            text=annotation,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font={"color": GREY, "size": 12},
        )
    return fig
