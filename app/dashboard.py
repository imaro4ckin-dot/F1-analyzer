import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dash
import dash_bootstrap_components as dbc

from app.layout import build_layout
from app.callbacks import register_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="F1 Stress Analyzer",
)

app.layout = build_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=False, port=8050, host="127.0.0.1")
