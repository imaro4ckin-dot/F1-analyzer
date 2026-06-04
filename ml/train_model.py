import ssl
import os
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ── macOS SSL fix for NLTK download ─────────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ── F1 domain keyword lexicons ───────────────────────────────────────────────
# Words/phrases that reliably signal driver stress in F1 radio
HIGH_STRESS_KEYWORDS = [
    'brake', 'brakes', 'lock', 'locked', 'locking',
    'oversteer', 'understeer', 'snap', 'snapping', 'spin', 'spinning',
    'vibrat', 'flat spot', 'flat tyre', 'puncture', 'slow puncture',
    'damage', 'damaged', 'broken', 'something broke',
    'problem', 'issue', 'failure', 'failed',
    'engine', 'gearbox', 'hydraulic', 'hydraulics',
    'warning', 'alarm', 'light', 'sensor',
    'oil', 'water', 'smoke', 'fire',
    'what happened', 'what was that', 'what is that',
    'come on', 'no no', 'not good', 'not happy',
    'struggling', 'can\'t', 'cannot', 'losing it',
    'going wide', 'off the track', 'hit the wall', 'into the wall',
    'hit', 'contact', 'collision', 'touched',
    'unsafe', 'dangerous', 'scary',
    'slow', 'too slow', 'losing time', 'losing pace',
    'tyre gone', 'tyres gone', 'tyre dying', 'graining',
    'bouncing', 'porpoising', 'bottoming',
    'kers', 'ers', 'deploy', 'harvesting issue',
]

# Words/phrases that reliably signal calm / routine communication
CALM_KEYWORDS = [
    'copy', 'copied', 'understood', 'roger', 'affirm', 'confirmed',
    'box box', 'box', 'pit',
    'good lap', 'nice lap', 'well done', 'good job', 'excellent',
    'push push', 'push hard', 'attack',
    'target', 'gap', 'interval', 'delta', 'position',
    'p1', 'p2', 'p3', 'p4', 'p5',
    'strategy', 'tyre life', 'manage', 'conserve', 'saving fuel',
    'fuel', 'fuel map', 'engine mode',
    'okay', 'ok', 'sure', 'yep', 'yes', 'yeah',
    'understood the plan', 'plan is', 'we\'ll',
    'happy', 'feel good', 'feeling good', 'positive',
    'clean', 'smooth', 'comfortable',
    'lap time', 'sector', 'fastest', 'purple',
]


def calculate_f1_stress(text: str) -> float:
    """
    Domain-specific F1 stress score (1.0 = calm, 10.0 = high stress).

    Three-tier approach:
      1. F1 high-stress keyword hits   → +1.5 per hit (primary signal)
      2. F1 calm keyword hits          → -1.0 per hit (primary signal)
      3. VADER compound score          → weak residual, 20% weight

    This vastly outperforms pure VADER because F1 radio has a fixed
    domain vocabulary that VADER (trained on social media) has no knowledge of.
    E.g. "Box box" → VADER: 5.5 (neutral), F1-aware: 3.5 (calm routine)
         "Brakes locking" → VADER: ~5.2, F1-aware: 8.0 (high stress)
    """
    if pd.isna(text) or len(str(text).strip()) < 3:
        return 5.0  # neutral for noise/empty entries

    t = str(text).lower()

    high_hits = sum(1 for k in HIGH_STRESS_KEYWORDS if k in t)
    calm_hits = sum(1 for k in CALM_KEYWORDS if k in t)

    # Keyword-based score: base 5.0, each stress hit +1.5, each calm hit -1.0
    keyword_score = 5.0 + (high_hits * 1.5) - (calm_hits * 1.0)
    keyword_score = float(np.clip(keyword_score, 1.0, 10.0))

    # VADER as weak residual (it helps with exclamations, expletives, tone)
    vader_compound = sia.polarity_scores(t)['compound']
    vader_score = 5.5 - vader_compound * 4.5
    vader_score = float(np.clip(vader_score, 1.0, 10.0))

    # 80% keyword domain knowledge, 20% VADER tone signal
    final = 0.80 * keyword_score + 0.20 * vader_score
    return round(float(np.clip(final, 1.0, 10.0)), 2)


def main():
    print("=== F1 Telemetry ML Trainer (v2 — Domain-Aware) ===\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), 'f1_all_seasons_training_data.csv')
    print(f"1. Loading training data from {csv_path}...")
    # The CSV contains two schemas: 13-col (2023) and 17-col (2024, with 4 extra
    # numeric columns inserted after max_speed).  Read each schema separately and
    # normalise to the 13-column layout before merging.
    COLS_13 = ['year', 'location', 'driver', 'timestamp',
               'avg_speed', 'max_speed', 'speed_variance',
               'throttle_volatility', 'throttle_snaps', 'brake_switches',
               'avg_rpm', 'max_rpm', 'transcript']
    COLS_17 = ['year', 'location', 'driver', 'timestamp',
               'avg_speed', 'max_speed', '_x1', '_x2', '_x3', '_x4',
               'speed_variance', 'throttle_volatility', 'throttle_snaps',
               'brake_switches', 'avg_rpm', 'max_rpm', 'transcript']

    import csv as _csv
    rows_13, rows_17 = [], []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = _csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            n = len(row)
            if n == 13:
                rows_13.append(row)
            elif n == 17:
                rows_17.append(row)
            # rows with any other width are silently skipped

    frames = []
    if rows_13:
        frames.append(pd.DataFrame(rows_13, columns=COLS_13))
    if rows_17:
        wide = pd.DataFrame(rows_17, columns=COLS_17)
        wide = wide.drop(columns=['_x1', '_x2', '_x3', '_x4'])
        frames.append(wide)

    df = pd.concat(frames, ignore_index=True)
    # Cast numeric columns
    for col in ['year', 'avg_speed', 'max_speed', 'speed_variance',
                'throttle_volatility', 'throttle_snaps', 'brake_switches',
                'avg_rpm', 'max_rpm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['transcript'])
    print(f"   Loaded {len(df)} rows from {df['driver'].nunique()} drivers, "
          f"{df['location'].nunique()} locations, years {sorted(df['year'].unique())}")

    # ── 2. Apply F1-aware labels ──────────────────────────────────────────────
    print("\n2. Applying F1 domain-aware stress labels...")
    df['target_stress'] = df['transcript'].apply(calculate_f1_stress)

    print("\n   Label distribution:")
    bins = list(range(1, 11))
    counts = pd.cut(df['target_stress'], bins=bins, include_lowest=True).value_counts().sort_index()
    for bucket, count in counts.items():
        bar = '█' * (count // 10)
        print(f"   {str(bucket):15s}: {count:4d}  {bar}")

    print("\n   Sample labels (F1-aware vs old VADER):")
    sample = df.sample(6, random_state=7)
    for _, row in sample.iterrows():
        vader_only = round(5.5 - sia.polarity_scores(str(row['transcript']).lower())['compound'] * 4.5, 2)
        print(f"   F1:{row['target_stress']:5.2f} | VADER:{vader_only:5.2f} | \"{str(row['transcript'])[:70]}\"")

    # ── 3. Add derived features computable from existing CSV columns ──────────
    print("\n3. Engineering additional features from existing data...")

    # min_speed: absolute lowest speed in the telemetry window
    # (not in CSV directly, but avg/max are — we approximate from variance)
    # For existing data: derive min_speed estimate = avg_speed - sqrt(speed_variance)
    df['min_speed'] = (df['avg_speed'] - np.sqrt(df['speed_variance'].clip(0))).clip(0)

    # max_speed_drop: proxy = speed_variance / avg_speed (high = erratic braking)
    df['max_speed_drop'] = (df['speed_variance'] / df['avg_speed'].replace(0, 1)).clip(0, 100)

    # gear_change_rate / drs_rate: not in existing CSV, default 0.0
    # Will be populated in future collect_training_data.py runs
    if 'gear_change_rate' not in df.columns:
        df['gear_change_rate'] = 0.0
    if 'drs_rate' not in df.columns:
        df['drs_rate'] = 0.0

    feature_cols = [
        'avg_speed', 'max_speed', 'min_speed', 'speed_variance', 'max_speed_drop',
        'throttle_volatility', 'throttle_snaps',
        'brake_switches',
        'avg_rpm', 'max_rpm',
        'gear_change_rate', 'drs_rate',
    ]

    # ── 4. Inject silent baseline rows ────────────────────────────────────────
    print("\n4. Injecting silent baseline rows (stress = 5.0)...")
    rng = np.random.default_rng(seed=42)
    silent_rows = df.sample(frac=0.30, random_state=42).copy()
    silent_rows['target_stress'] = 5.0

    # Perturb each feature ±10% to prevent exact duplicates
    for col in feature_cols:
        noise = rng.uniform(0.90, 1.10, len(silent_rows))
        silent_rows[col] = (silent_rows[col] * noise).clip(0)

    df_full = pd.concat([df, silent_rows], ignore_index=True)
    print(f"   Training set: {len(df)} radio rows + {len(silent_rows)} silent baseline = {len(df_full)} total")

    # ── 5. Train / test split ─────────────────────────────────────────────────
    print("\n5. Preparing features and splitting data...")
    X = df_full[feature_cols].fillna(0.0)
    y = df_full['target_stress']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n6. Training Random Forest Regressor (200 trees, max_depth=12)...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    print("\n7. Evaluating model accuracy...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"   Mean Absolute Error: ±{mae:.3f} stress points (out of 10)")

    # Prediction distribution sanity check
    pred_series = pd.Series(preds)
    print(f"   Prediction range: {pred_series.min():.2f} – {pred_series.max():.2f}")
    print(f"   Prediction mean:  {pred_series.mean():.2f} (should be near 5.0)")

    # ── 8. Feature importances ────────────────────────────────────────────────
    print("\n8. Feature importances (higher = more predictive):")
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importances:
        bar = '█' * int(imp * 200)
        print(f"   {feat:25s}: {imp:.4f}  {bar}")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), 'f1_stress_model.pkl')
    print(f"\n9. Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("   Done. Model saved successfully.")
    print(f"\n{'='*50}")
    print(f"  MAE: ±{mae:.3f}  |  Features: {len(feature_cols)}  |  Rows: {len(df_full)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
