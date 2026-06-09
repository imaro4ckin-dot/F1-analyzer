import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output

from app.layout import build_layout, build_driver_page
from app.callbacks import register_callbacks
from app.race_page import build_race_layout, register_race_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="F1 Stress Analyzer",
)

app.layout = build_layout()
register_callbacks(app)
register_race_callbacks(app)

# Expose WSGI server for Gunicorn: gunicorn app.dashboard:server
server = app.server


# ── Page routing ──────────────────────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/race":
        return build_race_layout()
    return build_driver_page()


# ── Active nav-tab highlighting ───────────────────────────────────────────────
from app.layout import RED, CARD2, BORDER, MUTED, WHITE, FONT_SANS

@app.callback(
    Output("nav-tab-driver", "style"),
    Output("nav-tab-race",   "style"),
    Input("url", "pathname"),
)
def highlight_nav_tab(pathname):
    base = {
        "color": MUTED,
        "fontSize": "10px",
        "fontWeight": "600",
        "letterSpacing": "0.18em",
        "fontFamily": FONT_SANS,
        "textTransform": "uppercase",
        "textDecoration": "none",
        "padding": "6px 14px",
        "borderRadius": "6px",
        "border": f"1px solid transparent",
        "transition": "all 0.2s",
        "whiteSpace": "nowrap",
    }
    active = {
        **base,
        "color": WHITE,
        "backgroundColor": CARD2,
        "border": f"1px solid {BORDER}",
    }
    if pathname == "/race":
        return base, active
    return active, base


if __name__ == "__main__":
    debug = os.getenv("DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", "8050"))
    host = os.getenv("HOST", "127.0.0.1")
    app.run(debug=debug, port=port, host=host)
