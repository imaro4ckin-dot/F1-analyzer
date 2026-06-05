import os
import numpy as np
import pandas as pd
import joblib

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "f1_stress_model.pkl")

# 12 features — must match train_model.py exactly
_FEATURES = [
    "avg_speed", "max_speed", "min_speed", "speed_variance", "max_speed_drop",
    "throttle_volatility", "throttle_snaps",
    "brake_switches",
    "avg_rpm", "max_rpm",
    "gear_change_rate", "drs_rate",
]


def load_model(path: str = _MODEL_PATH):
    """Load the trained Random Forest stress model from disk."""
    return joblib.load(path)


def extract_features(tel_slice: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 12 kinematic features from a telemetry window.
    Gracefully handles missing nGear / DRS columns (older data).
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
    speed_diff = speed.diff()

    speed_var = speed.var()
    throttle_vol = throttle_diff.mean()

    # min_speed: lowest speed in window — catches braking zone floors
    min_spd = float(speed.min())

    # max_speed_drop: single largest speed decrease — flags lockups / lifts
    neg_drops = speed_diff[speed_diff < 0]
    max_drop = float(abs(neg_drops.min())) if len(neg_drops) > 0 else 0.0

    # gear_change_rate: gear shifts per sample — erratic under stress
    if "nGear" in tel_slice.columns:
        gear_diff = tel_slice["nGear"].diff().abs()
        gear_changes = int(gear_diff[gear_diff > 0].count())
        gear_change_rate = round(gear_changes / max(len(tel_slice), 1), 4)
    else:
        gear_change_rate = 0.0

    # drs_rate: fraction of window with DRS open (8 = open in FastF1)
    if "DRS" in tel_slice.columns:
        drs_open = int((tel_slice["DRS"] == 8).sum())
        drs_rate = round(drs_open / max(len(tel_slice), 1), 4)
    else:
        drs_rate = 0.0

    def _safe(val):
        return 0.0 if (val is None or np.isnan(val)) else round(float(val), 4)

    return pd.DataFrame([{
        "avg_speed":          _safe(speed.mean()),
        "max_speed":          _safe(speed.max()),
        "min_speed":          _safe(min_spd),
        "speed_variance":     _safe(speed_var),
        "max_speed_drop":     _safe(max_drop),
        "throttle_volatility":_safe(throttle_vol),
        "throttle_snaps":     int(throttle_diff[throttle_diff > 20].count()),
        "brake_switches":     int(brake_diff[brake_diff > 0].count()),
        "avg_rpm":            _safe(rpm.mean()),
        "max_rpm":            _safe(rpm.max()),
        "gear_change_rate":   gear_change_rate,
        "drs_rate":           drs_rate,
    }])


def predict_continuous(
    tel: pd.DataFrame,
    model,
    window: int = 100,
    step: int = 10,
    use_anomaly: bool = False,
) -> pd.Series:
    """
    Predict stress across the full telemetry dataset.

    If use_anomaly=True (no radio available), falls back to the session-relative
    z-score method which requires no training labels and catches genuine anomalies
    relative to THIS driver's own baseline in THIS race.

    Otherwise uses the trained RF model.
    """
    if use_anomaly:
        return predict_anomaly_zscore(tel, window=window, step=step)

    n = len(tel)
    scores = pd.Series(index=tel.index, dtype=float)

    for i in range(window, n, step):
        tel_slice = tel.iloc[i - window: i]
        features = extract_features(tel_slice)
        raw = float(model.predict(features)[0])
        center = i - step // 2
        if 0 <= center < n:
            scores.iloc[center] = float(np.clip(round(raw, 2), 1.0, 10.0))

    return scores.ffill().bfill()


def predict_at_timestamp(
    tel: pd.DataFrame,
    radio_time: pd.Timestamp,
    model,
    window: int = 100,
) -> float:
    """
    Predict stress using the `window` rows immediately before `radio_time`.
    Used to annotate individual radio events on the timeline.
    """
    idx = (tel["Date"] - radio_time).abs().idxmin()
    loc = tel.index.get_loc(idx)
    start = max(0, loc - window)
    tel_slice = tel.iloc[start: loc + 1]
    features = extract_features(tel_slice)
    raw = float(model.predict(features)[0])
    return float(np.clip(round(raw, 2), 1.0, 10.0))


def predict_anomaly_zscore(
    tel: pd.DataFrame,
    window: int = 100,
    step: int = 10,
) -> pd.Series:
    """
    Session-relative anomaly detection. Requires no training data or radio.

    Slides a window across the telemetry, computes 12 features per window,
    then z-scores each window against the session's own mean/std.
    High z-score = this moment is abnormal relative to how THIS driver
    drove for the rest of THIS race.

    Score mapping:
      z ≈ 0.0  →  5.0  (baseline / normal driving)
      z ≈ 1.0  →  6.5  (slightly above average)
      z ≈ 2.0  →  8.0  (2 standard deviations out — clear anomaly)
      z ≥ 3.0  →  10.0 (clipped)
    """
    n = len(tel)
    feature_rows = []
    window_indices = []

    # Pass 1: compute features for every window
    for i in range(window, n, step):
        tel_slice = tel.iloc[i - window: i]
        feat = extract_features(tel_slice)
        feature_rows.append(feat.iloc[0].to_dict())
        window_indices.append(i - step // 2)

    if not feature_rows:
        return pd.Series(5.0, index=tel.index)

    feat_df = pd.DataFrame(feature_rows)

    # Pass 2: z-score each feature against session statistics
    means = feat_df.mean()
    stds = feat_df.std().replace(0.0, 1.0)   # avoid /0 on constant columns
    z_df = (feat_df - means) / stds

    # Mean absolute z-score across all features per window
    mean_abs_z = z_df.abs().mean(axis=1)

    # Map to 1–10 scale: baseline 5.0, each unit of z adds 1.5
    stress_scores = (5.0 + mean_abs_z * 1.5).clip(1.0, 10.0).round(2)

    # Build full-length Series aligned to telemetry index
    scores = pd.Series(index=tel.index, dtype=float)
    for idx, score in zip(window_indices, stress_scores):
        if 0 <= idx < n:
            scores.iloc[idx] = float(score)

    return scores.ffill().bfill()
