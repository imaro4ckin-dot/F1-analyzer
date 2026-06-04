import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import ssl

# --- THE FIX: Bypass macOS SSL certificate verification for NLTK ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 1. Download the NLP sentiment dictionary
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()


def calculate_nlp_stress(text):
    """
    Converts a transcript into a stress score from 1.0 (Calm) to 10.0 (High Stress/Anger)
    """
    if pd.isna(text):
        return 5.0  # Neutral baseline if transcript is missing

    # VADER compound score returns a value between -1.0 (Negative/Angry) and +1.0 (Positive/Calm).
    score = sia.polarity_scores(str(text))['compound']

    # Map VADER's [-1.0 to 1.0] scale to our F1 Stress scale [10.0 to 1.0]
    stress_score = 5.5 - (score * 4.5)
    return round(stress_score, 2)


def main():
    print("=== F1 Telemetry ML Trainer ===")

    print("\n1. Loading raw training data...")
    df = pd.read_csv('f1_all_seasons_training_data.csv')

    print("2. Running NLP Sentiment Analysis to generate 'Ground Truth' labels...")
    df['target_stress'] = df['transcript'].apply(calculate_nlp_stress)

    print("\n--- Sample NLP Grading ---")
    sample = df[['transcript', 'target_stress']].sample(5)
    for index, row in sample.iterrows():
        print(f"[{row['target_stress']}/10.0] - \"{row['transcript'][:80]}...\"")

    print("\n3. Preparing Machine Learning Features...")
    features = ['avg_speed', 'max_speed', 'speed_variance', 'throttle_volatility',
                'throttle_snaps', 'brake_switches', 'avg_rpm', 'max_rpm']
    X = df[features]
    y = df['target_stress']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("4. Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    print("\n5. Evaluating Model Accuracy...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print(f"-> Mean Absolute Error: ±{mae:.2f} stress points (out of 10)")

    print("\n6. Saving Model to Disk...")
    joblib.dump(model, "f1_stress_model.pkl")
    print("-> Model successfully saved as 'f1_stress_model.pkl'!")


if __name__ == "__main__":
    main()