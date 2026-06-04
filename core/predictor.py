import os
import numpy as np
import pandas as pd
import joblib

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "f1_stress_model.pkl")

_FEATURES = [
    "avg_speed", "max_speed", "speed_variance",
    "throttle_volatility", "throttle_snaps",
    "brake_switches", "avg_rpm", "max_rpm",
]


def load_model(path: str = _MODEL_PATH):
    """Load the trained Random Forest stress model from disk."""
    return joblib.load(path)


def extract_features(tel_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 8 kinematic features from a telemetry window.
    Returns a single-row DataFrame ready for model.predict().
    """
    if tel_slice.empty or len(tel_slice) < 5:
        return pd.DataFrame([{f: 0.0 for f in _FEATURES}])

    speed = tel_slice["Speed"]
    throttle = tel_slice["Throttle"]
    brake = tel_slice["Brake"].astype(int)
    rpm = tel_slice["RPM"]

    throttle_diff = throttle.diff().abs()
    brake_diff = brake.diff().abs()

    speed_var = speed.var()
    throttle_vol = throttle_diff.mean()

    return pd.DataFrame([{
        "avg_speed": round(float(speed.mean()), 2),
        "max_speed": int(speed.max()),
        "speed_variance": round(float(speed_var), 2) if not np.isnan(speed_var) else 0.0,
        "throttle_volatility": round(float(throttle_vol), 2) if not np.isnan(throttle_vol) else 0.0,
        "throttle_snaps": int(throttle_diff[throttle_diff > 20].count()),
        "brake_switches": int(brake_diff[brake_diff > 0].count()),
        "avg_rpm": round(float(rpm.mean()), 1),
        "max_rpm": int(rpm.max()),
    }])


def predict_continuous(tel: pd.DataFrame, model, window: int = 100, step: int = 10) -> pd.Series:
    """
    Slide a window of `window` rows over the full telemetry DataFrame and predict stress
    at each step. Returns a Series indexed identically to `tel`, forward-filled between steps.

    This produces a continuous stress curve for the entire race/session.
    """
    n = len(tel)
    scores = pd.Series(index=tel.index, dtype=float)

    for i in range(window, n, step):
        tel_slice = tel.iloc[i - window: i]
        features = extract_features(tel_slice)
        raw = model.predict(features)[0]
        clamped = float(min(10.0, max(1.0, round(raw, 2))))
        # Assign to the centre of the window
        scores.iloc[i - step // 2] = clamped

    # Forward-fill gaps, then back-fill the leading NaN region
    scores = scores.ffill().bfill()
    return scores


def predict_at_timestamp(tel: pd.DataFrame, radio_time: pd.Timestamp, model,
                         window: int = 100) -> float:
    """
    Predict stress using the `window` rows of telemetry immediately before `radio_time`.
    Used to annotate individual radio events.
    """
    idx_pos = (tel["Date"] - radio_time).abs().argsort().iloc[0]
    idx = tel.index[idx_pos]
    loc = tel.index.get_loc(idx)
    start = max(0, loc - window)
    tel_slice = tel.iloc[start: loc + 1]
    features = extract_features(tel_slice)
    raw = model.predict(features)[0]
    return float(min(10.0, max(1.0, round(raw, 2))))
