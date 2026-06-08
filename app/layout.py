from dash import dcc, html

# ── Design tokens (mirrored in assets/style.css :root) ───────────────────────
BG        = "#070709"        # near-black base
SURFACE   = "#0E0E12"        # navbar / elevated surface
CARD      = "#12121A"        # panel cards
CARD2     = "#1A1A26"        # inner elevated (dropdowns, inputs)
BORDER    = "rgba(255,255,255,0.06)"   # hairline card border
BORDER2   = "rgba(255,255,255,0.12)"   # visible divider
RED       = "#E8000D"        # F1 accent
RED_GLOW  = "rgba(232,0,13,0.25)"
RED_DIM   = "#7A0300"
WHITE     = "#EDEDEF"        # primary text
MUTED     = "#8A8F98"        # secondary text
MUTED2    = "#4A4D55"        # disabled / placeholder
GREEN     = "#00D264"        # positive
ORANGE    = "#FF8C00"        # warning
YELLOW    = "#FFD700"        # P1 gold
TEAL      = "#00D4FF"        # data accent

# Tyre compound colours (used by callbacks)
TYRE_SOFT   = "#E8000D"
TYRE_MEDIUM = "#FFF200"
TYRE_HARD   = "#EDEDEF"

# ── Typography ───────────────────────────────────────────────────────────────
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace"
FONT_SANS = "'Inter', 'Fira Sans', system-ui, sans-serif"

# ── Legacy aliases kept so callbacks don't break ─────────────────────────────
GREY  = MUTED
GREY2 = MUTED2

# ── Shared style helpers ─────────────────────────────────────────────────────
LABEL_STYLE = {
    "color": MUTED,
    "fontSize": "9px",
    "letterSpacing": "0.2em",
    "textTransform": "uppercase",
    "marginBottom": "6px",
    "fontFamily": FONT_SANS,
    "fontWeight": "600",
    "lineHeight": "1",
}

SECTION_TITLE_STYLE = {
    **LABEL_STYLE,
    "fontSize": "8px",
    "letterSpacing": "0.25em",
}

DROPDOWN_STYLE = {
    "backgroundColor": CARD2,
    "color": RED ,
    "border": f"1px solid {BORDER2}",
    "borderRadius": "8px",
    "fontFamily": FONT_MONO,
    "fontSize": "12px",
}

INPUT_STYLE = {
    **DROPDOWN_STYLE,
    "padding": "9px 14px",
    "outline": "none",
    "textTransform": "uppercase",
    "letterSpacing": "0.12em",
    "height": "38px",
    "boxSizing": "border-box",
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
    return html.Div(
        id=chip_id,
        className="stat-chip",
        style={
            "backgroundColor": CARD2,
            "border": f"1px solid {BORDER}",
            "borderRadius": "12px",
            "padding": "12px 20px",
            "minWidth": "120px",
            "flex": "1",
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            "position": "relative",
            "overflow": "hidden",
        },
        children=[
            html.Span(label, style=LABEL_STYLE),
            html.Div("—", id=f"{chip_id}-val", style={
                "color": WHITE,
                "fontSize": "18px",
                "fontWeight": "700",
                "fontFamily": FONT_MONO,
                "letterSpacing": "0.03em",
                "lineHeight": "1",
            }),
        ],
    )


def _card(children, flex=None, min_width=None, extra_style=None):
    style = {
        "backgroundColor": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "16px",
        "padding": "20px",
    }
    if flex:
        style["flex"] = flex
    if min_width:
        style["minWidth"] = min_width
    if extra_style:
        style.update(extra_style)
    return html.Div(className="f1-card", style=style, children=children)


def _section_header(title: str, badge: str = ""):
    return html.Div(
        className="section-header",
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "marginBottom": "16px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Div(style={
                        "width": "6px", "height": "6px",
                        "borderRadius": "50%",
                        "backgroundColor": RED,
                        "boxShadow": f"0 0 8px {RED_GLOW}",
                        "flexShrink": "0",
                    }),
                    html.Span(title, style={
                        "color": WHITE,
                        "fontSize": "11px",
                        "fontWeight": "600",
                        "letterSpacing": "0.06em",
                        "fontFamily": FONT_SANS,
                        "textTransform": "uppercase",
                    }),
                ],
            ),
            html.Span(badge, style={
                "color": MUTED,
                "fontSize": "9px",
                "letterSpacing": "0.12em",
                "fontFamily": FONT_MONO,
                "textTransform": "uppercase",
                "backgroundColor": "rgba(255,255,255,0.04)",
                "border": f"1px solid {BORDER}",
                "borderRadius": "4px",
                "padding": "3px 8px",
            }) if badge else None,
        ],
    )


def _empty_figure(annotation: str = ""):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"color": WHITE, "family": FONT_MONO},
        margin={"l": 44, "r": 16, "t": 8, "b": 36},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "color": MUTED},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "color": MUTED},
    )
    if annotation:
        fig.add_annotation(
            text=annotation.replace("\n", "<br>"),
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font={"color": MUTED2, "size": 12, "family": FONT_SANS},
            align="center",
        )
    return fig


def build_layout():
    return html.Div(
        id="app-root",
        style={
            "backgroundColor": BG,
            "minHeight": "100vh",
            "fontFamily": FONT_SANS,
            "color": WHITE,
        },
        children=[

            # ── Navbar ───────────────────────────────────────────────────────
            html.Header(
                id="navbar",
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"1px solid {BORDER}",
                    "padding": "0 32px",
                    "height": "56px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "position": "sticky",
                    "top": "0",
                    "zIndex": "200",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "16px"},
                        children=[
                            # Chequered flag grid mark
                            html.Div(
                                className="nav-logo",
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(4, 5px)",
                                    "gridTemplateRows": "repeat(3, 5px)",
                                    "gap": "1px",
                                },
                                children=[
                                    html.Div(style={"backgroundColor": c})
                                    for c in [
                                        WHITE, RED,   WHITE, RED,
                                        RED,   WHITE, RED,   WHITE,
                                        WHITE, RED,   WHITE, RED,
                                    ]
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "flexDirection": "column", "gap": "3px"},
                                children=[
                                    html.Span("F1 STRESS ANALYZER", style={
                                        "fontSize": "14px",
                                        "fontWeight": "700",
                                        "letterSpacing": "0.16em",
                                        "color": WHITE,
                                        "fontFamily": FONT_SANS,
                                        "lineHeight": "1",
                                    }),
                                    html.Span("AI · TELEMETRY · RADIO", style={
                                        "fontSize": "8px",
                                        "letterSpacing": "0.22em",
                                        "color": MUTED,
                                        "fontFamily": FONT_MONO,
                                        "lineHeight": "1",
                                    }),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id="status-msg",
                        style={
                            "color": MUTED,
                            "fontSize": "10px",
                            "letterSpacing": "0.1em",
                            "fontFamily": FONT_MONO,
                            "maxWidth": "380px",
                            "textAlign": "right",
                            "lineHeight": "1.5",
                        },
                    ),
                ],
            ),

            # ── Controls bar ─────────────────────────────────────────────────
            html.Div(
                id="controls-panel",
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
                        _dropdown(
                            "dd-year", "Year",
                            options=[{"label": str(y), "value": y} for y in [2023, 2024, 2025]],
                            value=2024,
                        ),
                    ], style={"width": "100px"}),

                    html.Div([
                        html.Div("Grand Prix", style=LABEL_STYLE),
                        _dropdown("dd-race", "Select race…"),
                    ], style={"minWidth": "220px", "flex": "2"}),

                    html.Div(className="ctrl-divider", style={
                        "width": "1px", "height": "38px",
                        "backgroundColor": BORDER2,
                        "alignSelf": "flex-end",
                        "marginBottom": "1px",
                    }),

                    html.Div([
                        html.Div("Driver", style=LABEL_STYLE),
                        dcc.Input(
                            id="input-driver",
                            type="text", value="LEC", maxLength=3,
                            placeholder="LEC",
                            style={**INPUT_STYLE, "width": "76px"},
                        ),
                    ]),

                    html.Span("VS", style={
                        "color": MUTED2, "fontSize": "9px",
                        "letterSpacing": "0.2em", "fontFamily": FONT_SANS,
                        "fontWeight": "600", "alignSelf": "flex-end",
                        "paddingBottom": "11px",
                    }),

                    html.Div([
                        html.Div("Compare", style=LABEL_STYLE),
                        dcc.Input(
                            id="input-driver2",
                            type="text", value="", maxLength=3,
                            placeholder="optional",
                            style={**INPUT_STYLE, "width": "100px"},
                        ),
                    ]),

                    html.Div(className="ctrl-divider", style={
                        "width": "1px", "height": "38px",
                        "backgroundColor": BORDER2,
                        "alignSelf": "flex-end",
                        "marginBottom": "1px",
                    }),

                    html.Div([
                        html.Div("\u00a0", style={**LABEL_STYLE, "visibility": "hidden"}),
                        html.Button(
                            id="btn-analyze",
                            n_clicks=0,
                            className="btn-analyze",
                            children=html.Span("ANALYZE", style={
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
                ],
            ),

            # ── Stats bar (hidden until first analysis) ───────────────────────
            html.Div(
                id="stats-bar",
                style={
                    "backgroundColor": SURFACE,
                    "borderBottom": f"1px solid {BORDER}",
                    "padding": "12px 32px",
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

            # ── Main body ─────────────────────────────────────────────────────
            html.Main(
                style={"padding": "24px 32px 0 32px"},
                children=[

                    # Row 1: Telemetry + Track map
                    html.Div(
                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                        children=[
                            _card(
                                flex="6", min_width="460px",
                                children=[
                                    _section_header(
                                        "Telemetry & AI Stress",
                                        badge="speed · throttle · brake · drs",
                                    ),
                                    dcc.Graph(
                                        id="chart-telemetry",
                                        config={
                                            "displayModeBar": True,
                                            "modeBarButtonsToRemove": [
                                                "select2d", "lasso2d", "autoScale2d",
                                                "hoverClosestCartesian",
                                                "hoverCompareCartesian",
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
                            _card(
                                flex="4", min_width="300px",
                                children=[
                                    _section_header(
                                        "Track Map",
                                        badge="click marker → radio",
                                    ),
                                    dcc.Graph(
                                        id="chart-track",
                                        config={
                                            "displayModeBar": True,
                                            "modeBarButtonsToRemove": [
                                                "select2d", "lasso2d",
                                                "toggleSpikelines",
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

                    # Radio panel
                    html.Div(
                        id="radio-panel",
                        style={"marginTop": "16px"},
                        children=[],
                    ),

                    # Row 2: Lap stress + Lap times
                    html.Div(
                        style={
                            "display": "flex", "gap": "16px",
                            "flexWrap": "wrap", "marginTop": "16px",
                        },
                        children=[
                            _card(
                                flex="1", min_width="280px",
                                children=[
                                    _section_header(
                                        "Lap-by-Lap Stress",
                                        badge="avg ai score per lap",
                                    ),
                                    dcc.Graph(
                                        id="chart-lap-stress",
                                        config={"displayModeBar": False, "displaylogo": False},
                                        style={"height": "180px"},
                                        figure=_empty_figure(""),
                                    ),
                                ],
                            ),
                            _card(
                                flex="1", min_width="280px",
                                children=[
                                    _section_header(
                                        "Lap Time Evolution",
                                        badge="seconds per lap",
                                    ),
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

                    # Leaderboard
                    html.Div(
                        id="leaderboard-panel",
                        style={"marginTop": "16px", "paddingBottom": "48px"},
                        children=[],
                    ),
                ],
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
