import fastf1
import requests
import pandas as pd
import os
import whisper
import tempfile
import ssl
import joblib
import warnings

# Ignore sklearn warnings about feature names to keep the console clean
warnings.filterwarnings('ignore')

# 1. Bypass macOS SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Setup FastF1 Cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# 3. Load the AI Models
print("Loading Whisper AI Model (Medium)...")
model = whisper.load_model("medium")

print("Loading Kinematic Stress ML Model...")
try:
    stress_model = joblib.load("f1_stress_model.pkl")
except FileNotFoundError:
    print("Error: f1_stress_model.pkl not found! Please run train_model.py first.")
    exit()


# --- THE ML FALLBACK FUNCTION ---
def predict_ml_stress(telemetry_slice):
    """Uses the trained Random Forest model to predict stress from kinematics."""
    if telemetry_slice.empty or len(telemetry_slice) < 5:
        return 5.0  # Return baseline neutral score if slice is too small

    avg_speed = telemetry_slice['Speed'].mean()
    max_speed = telemetry_slice['Speed'].max()
    speed_var = telemetry_slice['Speed'].var()

    throttle_diff = telemetry_slice['Throttle'].diff().abs()
    throttle_volatility = throttle_diff.mean()
    throttle_snaps = throttle_diff[throttle_diff > 20].count()

    brake_diff = telemetry_slice['Brake'].astype(int).diff().abs()
    brake_switches = brake_diff[brake_diff > 0].count()

    avg_rpm = telemetry_slice['RPM'].mean()
    max_rpm = telemetry_slice['RPM'].max()

    # Structure the features exactly how the model was trained
    features = pd.DataFrame([{
        "avg_speed": round(avg_speed, 2),
        "max_speed": int(max_speed),
        "speed_variance": round(speed_var, 2) if not pd.isna(speed_var) else 0.0,
        "throttle_volatility": round(throttle_volatility, 2) if not pd.isna(throttle_volatility) else 0.0,
        "throttle_snaps": int(throttle_snaps),
        "brake_switches": int(brake_switches),
        "avg_rpm": round(avg_rpm, 1),
        "max_rpm": int(max_rpm)
    }])

    # Ask the ML model to predict the stress
    predicted_stress = stress_model.predict(features)[0]
    # Cap it between 1.0 and 10.0 just in case of weird outliers
    return min(10.0, max(1.0, round(predicted_stress, 2)))


# ==========================================
# === 1. INTERACTIVE RACE SELECTION ======
# ==========================================
print("\n=== F1 Telemetry & Radio Analyzer ===")
year = input("Enter the Year (e.g., 2023, 2024): ").strip()

print(f"\nFetching available races for {year} from OpenF1...")

if int(year) < 2023:
    print(f"\n[Data Limitation]: The OpenF1 API only contains historical data from 2023 onwards.")
    exit()

sessions_url = f"https://api.openf1.org/v1/sessions?year={year}&session_name=Race"
races_data = requests.get(sessions_url).json()

if isinstance(races_data, dict) or not races_data:
    print(f"\n[API Error]: Failed to fetch races for {year}.")
    exit()

print("\n--- Available Races ---")
for i, race in enumerate(races_data):
    print(f"{i + 1}. {race['country_name']} - {race['location']}")

race_choice = int(input(f"\nSelect a race number (1-{len(races_data)}): ").strip()) - 1
selected_race = races_data[race_choice]

session_key = selected_race['session_key']
fastf1_location = selected_race['location']

print(f"\nLoading FastF1 Session Data for {year} {fastf1_location}...")
session = fastf1.get_session(int(year), fastf1_location, 'R')
session.load(telemetry=True, weather=False)

# ==========================================
# === 2. INTERACTIVE DRIVER SELECTION ====
# ==========================================
driver_map = {}
for num in session.drivers:
    try:
        drv_info = session.get_driver(num)
        if drv_info['Abbreviation']:
            driver_map[drv_info['Abbreviation']] = num
    except Exception:
        continue

print("\n--- Available Drivers ---")
print(", ".join(driver_map.keys()))

selected_code = input("\nEnter the 3-letter code of the driver to analyze: ").strip().upper()

if selected_code not in driver_map:
    fallback = list(driver_map.keys())[0]
    print(f"Error: '{selected_code}' is not valid. Defaulting to {fallback}.")
    selected_code = fallback

driver_num = driver_map[selected_code]
print(f"\nExtracting data for {selected_code} (Car #{driver_num})...")

tel_driver = session.laps.pick_drivers(selected_code).get_car_data()

# ==========================================
# === 3. RADIO & TELEMETRY ENGINE ======
# ==========================================
print(f"Fetching OpenF1 Radio Data for {selected_code}...")
url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}&driver_number={driver_num}"

has_radio = True
try:
    response = requests.get(url, timeout=10)
    radio_messages = response.json()
    if isinstance(radio_messages, dict) or not radio_messages:
        has_radio = False
except Exception:
    has_radio = False

# --- CONTINUOUS EXECUTION FLOW (ML FALLBACK) ---
if not has_radio:
    print(f"\n[Radio Feed Missing]: OpenF1 has no radio logs on file for {selected_code} at {fastf1_location}.")
    print("Pivoting pipeline to Continuous AI Telemetry Analysis...\n")
    print(f"=== Continuous Telemetry & ML Stress Stream ({selected_code} - {fastf1_location}) ===")

    if tel_driver.empty or len(tel_driver) < 100:
        print("Error: No telemetry data available for this driver layout.")
        exit()

    step_size = 300
    window_size = 100

    for i in range(window_size, len(tel_driver), step_size):
        tele_row = tel_driver.iloc[i]
        telemetry_slice = tel_driver.iloc[i - window_size: i]

        timestamp = tele_row['Date']
        speed = tele_row['Speed']
        throttle = tele_row['Throttle']
        brake = tele_row['Brake']

        ml_stress = predict_ml_stress(telemetry_slice)

        print(f"Session Time: {timestamp}")
        print(f"Car State: {speed} km/h | Throttle: {throttle}% | Brake: {brake}%")
        print(f"AI Predicted Stress Level: {ml_stress}/10.0")
        print("-" * 40)

else:
    print(f"\n=== Radio, Telemetry & Transcription Sync ({selected_code} - {fastf1_location}) ===")

    for message in radio_messages[:15]:
        radio_time = pd.to_datetime(message['date']).tz_localize(None)

        closest_tele = tel_driver.iloc[(tel_driver['Date'] - radio_time).abs().argsort()[:1]]

        if closest_tele.empty:
            speed, throttle, brake = 0, 0, 0
            telemetry_slice = pd.DataFrame()
        else:
            speed = closest_tele['Speed'].values[0]
            throttle = closest_tele['Throttle'].values[0]
            brake = closest_tele['Brake'].values[0]
            idx = closest_tele.index[0]
            telemetry_slice = tel_driver.loc[max(0, idx - 100): idx]

        print(f"Time: {radio_time.time()}")
        print(f"Car State: {speed} km/h | Throttle: {throttle}% | Brake: {brake}%")

        if not telemetry_slice.empty:
            ml_stress = predict_ml_stress(telemetry_slice)
            print(f"AI Predicted Stress Level (Pre-Radio): {ml_stress}/10.0")

        audio_url = message['recording_url']
        print(f"Audio Feed: {audio_url}")

        try:
            audio_response = requests.get(audio_url, timeout=10)
            if len(audio_response.content) < 1000:
                print("Status: [Skipped - Audio file payload is empty or uninitialized]\n")
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(audio_response.content)
                temp_file_path = temp_audio.name

            print("Transcribing audio...")
            dynamic_prompt = f"Formula 1 team radio. {selected_code} speaking to race engineer. Terms: box, tyres, DRS, pit lane, braking, understeer, oversteer, apex, sector, lap time."

            result = model.transcribe(
                temp_file_path, fp16=False, language="en", temperature=0.0,
                condition_on_previous_text=False, no_speech_threshold=0.6,
                initial_prompt=dynamic_prompt
            )

            transcript = result['text'].strip()

            if "speaking to race engineer" in transcript or not transcript:
                print("Transcript: [No audible human speech detected - Engine static only]\n")
            else:
                print(f"Transcript: \"{transcript}\"\n")

            os.remove(temp_file_path)

        except Exception as e:
            print(f"Status: [Skipped due to unexpected exception: {e}]\n")