import fastf1
import requests
import pandas as pd
import os
import whisper
import tempfile
import ssl


# --- HELPER FUNCTION: Telemetry Fallback ---
def calculate_telemetry_agitation(telemetry_slice):
    """Calculates how erratic the driver's inputs are to estimate stress."""
    # 1. Throttle Volatility (Snapping the throttle)
    throttle_diff = telemetry_slice['Throttle'].diff().abs()
    snaps = throttle_diff[throttle_diff > 20].count()

    # 2. Brake Spikes (Erratic or micro-locking brakes)
    # astype(int) safely converts boolean (True/False) brake data into 1/0
    brake_diff = telemetry_slice['Brake'].astype(int).diff().abs()
    brake_stamps = brake_diff[brake_diff > 0].count()

    # 3. Calculate Score
    agitation_score = (snaps * 1.5) + (brake_stamps * 0.5)
    return min(10.0, agitation_score)


# 1. Bypass macOS SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Setup FastF1 Cache
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# 3. Load the Whisper AI Model
print("Loading Whisper AI Model...")
model = whisper.load_model("medium")

# ==========================================
# === 1. INTERACTIVE RACE SELECTION ======
# ==========================================
print("\n=== F1 Telemetry & Radio Analyzer ===")
year = input("Enter the Year (e.g., 2023, 2024): ").strip()

print(f"\nFetching available races for {year} from OpenF1...")

# Guardrail: OpenF1 only supports 2023 onwards
if int(year) < 2023:
    print(f"\n[Data Limitation]: The OpenF1 API only contains historical data from 2023 to the present day.")
    print("Please restart the script and select a year of 2023 or newer.")
    exit()

# Query OpenF1 for all 'Race' sessions in the given year
sessions_url = f"https://api.openf1.org/v1/sessions?year={year}&session_name=Race"
races_data = requests.get(sessions_url).json()

# Handle API Error Dictionaries (like {"detail": "Not found"}) or empty data
if isinstance(races_data, dict):
    error_msg = races_data.get('detail', 'Unknown API Error')
    print(f"\n[API Error]: {error_msg}")
    print(f"Failed to fetch races for {year}. The season may not have started yet.")
    exit()
elif not races_data:
    print(f"Error: No race data found for the year {year}. Exiting.")
    exit()

# Build the Race Menu
print("\n--- Available Races ---")
for i, race in enumerate(races_data):
    print(f"{i + 1}. {race['country_name']} - {race['location']}")

race_choice = int(input(f"\nSelect a race number (1-{len(races_data)}): ").strip()) - 1
selected_race = races_data[race_choice]

# Store the keys we need for both APIs
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
    drv_info = session.get_driver(num)
    driver_map[drv_info['Abbreviation']] = num

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
response = requests.get(url)
radio_messages = response.json()

# --- THE FIX: Telemetry Fallback Trigger ---
# If radio data is missing or corrupted, we skip the NLP engine and run the telemetry analysis
if isinstance(radio_messages, dict) or not radio_messages:
    if isinstance(radio_messages, dict):
        error_msg = radio_messages.get('detail', 'Unknown API Error')
        print(f"\n[API Error]: {error_msg}")
        print(f"OpenF1 currently has no radio data on file for {selected_code} at {fastf1_location}.")
    else:
        print(f"\nNo radio messages found for {selected_code} in this session.")

    print("\n[Radio Feed Missing] Falling back to Kinematic Agitation Analysis...")

    if not tel_driver.empty and len(tel_driver) > 500:
        # Take a 500-millisecond sample slice of the race (bypassing Lap 1 chaos)
        start_idx = min(2000, len(tel_driver) - 500)
        end_idx = start_idx + 500
        telemetry_slice = tel_driver.iloc[start_idx:end_idx]

        agitation = calculate_telemetry_agitation(telemetry_slice)
        print(f"Estimated Driver Stress based on physical inputs: {agitation}/10.0")
    else:
        print("Insufficient telemetry data to calculate fallback score.")

    print("\nExiting fallback analysis.")
    exit()

print(f"\n=== Radio, Telemetry & Transcription Sync ({selected_code} - {fastf1_location}) ===")

# Process the first 15 messages
for message in radio_messages[:15]:
    radio_time = pd.to_datetime(message['date']).tz_localize(None)

    # Telemetry alignment
    closest_tele = tel_driver.iloc[(tel_driver['Date'] - radio_time).abs().argsort()[:1]]

    if closest_tele.empty:
        speed, throttle, brake = 0, 0, 0
    else:
        speed = closest_tele['Speed'].values[0]
        throttle = closest_tele['Throttle'].values[0]
        brake = closest_tele['Brake'].values[0]

    print(f"Time: {radio_time.time()}")
    print(f"Car State: {speed} km/h | Throttle: {throttle}% | Brake: {brake}%")

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
            temp_file_path,
            fp16=False,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            initial_prompt=dynamic_prompt
        )

        transcript = result['text'].strip()

        # --- THE FIX: Catch Prompt Hallucinations ---
        if "speaking to race engineer" in transcript or not transcript:
            print("Transcript: [No audible human speech detected - Engine static only]\n")
        else:
            print(f"Transcript: \"{transcript}\"\n")

        os.remove(temp_file_path)

    except RuntimeError:
        print("Status: [Skipped - FFmpeg found unreadable, broken, or corrupt audio frames]\n")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    except Exception as e:
        print(f"Status: [Skipped due to unexpected exception: {e}]\n")