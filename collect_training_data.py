import fastf1
import requests
import pandas as pd
import os
import whisper
import tempfile
import ssl
import time

# --- SETUP & CONFIGURATION ---
# 1. Bypass macOS SSL certificate verification for downloading model weights
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Configure local directory caching for FastF1 data packages
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# OpenF1 data feeds began in 2023. We collect up to the current 2026 season.
TARGET_YEARS = [2023, 2024, 2025, 2026]
OUTPUT_CSV = "f1_all_seasons_training_data.csv"

print("Loading Whisper AI Model (Medium)...")
model = whisper.load_model("medium")


# --- FEATURE ENGINEERING HELPER ---
def extract_telemetry_features(telemetry_slice):
    """Extracts detailed statistical features from a telemetry slice before a radio call."""
    if telemetry_slice.empty or len(telemetry_slice) < 5:
        return None

    # Speed metrics
    avg_speed = telemetry_slice['Speed'].mean()
    max_speed = telemetry_slice['Speed'].max()
    speed_var = telemetry_slice['Speed'].var()

    # Throttle inputs ( volatility & snap frequency )
    throttle_diff = telemetry_slice['Throttle'].diff().abs()
    throttle_volatility = throttle_diff.mean()
    throttle_snaps = throttle_diff[throttle_diff > 20].count()

    # Brake inputs ( frequency of applications )
    brake_diff = telemetry_slice['Brake'].astype(int).diff().abs()
    brake_switches = brake_diff[brake_diff > 0].count()

    # Engine strain metrics
    avg_rpm = telemetry_slice['RPM'].mean()
    max_rpm = telemetry_slice['RPM'].max()

    return {
        "avg_speed": round(avg_speed, 2),
        "max_speed": int(max_speed),
        "speed_variance": round(speed_var, 2) if not pd.isna(speed_var) else 0.0,
        "throttle_volatility": round(throttle_volatility, 2) if not pd.isna(throttle_volatility) else 0.0,
        "throttle_snaps": int(throttle_snaps),
        "brake_switches": int(brake_switches),
        "avg_rpm": round(avg_rpm, 1),
        "max_rpm": int(max_rpm)
    }


# --- DATA PIPELINE ENGINE ---
def main():
    # Initialize persistent storage layer with structured schema matrix if missing
    if not os.path.exists(OUTPUT_CSV):
        headers = ["year", "location", "driver", "timestamp", "avg_speed", "max_speed",
                   "speed_variance", "throttle_volatility", "throttle_snaps",
                   "brake_switches", "avg_rpm", "max_rpm", "transcript"]
        pd.DataFrame(columns=headers).to_csv(OUTPUT_CSV, index=False)

    # Outer loop: Iterate sequentially across all available F1 data eras
    for target_year in TARGET_YEARS:
        print(f"\n>>>> STARTING DATABASE GENERATION FOR THE {target_year} SEASON <<<<")

        sessions_url = f"https://api.openf1.org/v1/sessions?year={target_year}&session_name=Race"
        try:
            races = requests.get(sessions_url, timeout=15).json()
            if isinstance(races, dict) or not races:
                print(f"Skipping season {target_year}: Could not retrieve valid session list.")
                continue
        except Exception as e:
            print(f"API Connection error for season {target_year}: {e}")
            continue

        print(f"Found {len(races)} races to analyze for the {target_year} championship.")

        # Mid loop: Step through each individual Grand Prix event
        for race_idx, race in enumerate(races):
            session_key = race['session_key']
            location = race['location']
            print(f"\n=============================================")
            print(f"PROCESSING GRAND PRIX: [{race_idx + 1}/{len(races)}] {target_year} {location}")
            print(f"=============================================")

            # Load FastF1 core data matrix
            try:
                ff1_session = fastf1.get_session(target_year, location, 'R')
                ff1_session.load(telemetry=True, weather=False)
            except Exception as e:
                print(f"Skipping race layout: FastF1 could not load telemetry for {location}. Error: {e}")
                continue

            # Dynamically fetch ALL drivers who hit the track for this race grid
            session_drivers = []
            for num in ff1_session.drivers:
                try:
                    drv_info = ff1_session.get_driver(num)
                    if drv_info['Abbreviation']:
                        session_drivers.append(drv_info['Abbreviation'])
                except Exception:
                    continue

            # Remove any duplicate mappings to keep iterations pure
            session_drivers = sorted(list(set(session_drivers)))
            print(f"Detected {len(session_drivers)} active drivers on the grid.")

            # Inner loop: Dynamically process every driver on the grid layout
            for driver in session_drivers:
                print(f" -> Mining logs for {driver}...")

                try:
                    drv_num = ff1_session.get_driver(driver)['DriverNumber']
                    tel_driver = ff1_session.laps.pick_drivers(driver).get_car_data()
                    if tel_driver.empty:
                        continue
                except Exception:
                    continue

                # Query OpenF1 live timing database for raw team radio logs
                radio_url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}&driver_number={drv_num}"
                try:
                    radio_messages = requests.get(radio_url, timeout=10).json()
                    if isinstance(radio_messages, dict) or not radio_messages:
                        continue  # No radio data for this specific driver at this track layout
                except Exception:
                    continue

                collected_rows = []

                # Sift through radio records (cap sample variance to top 25 records per driver to maintain class balance)
                for msg in radio_messages[:25]:
                    msg_time = pd.to_datetime(msg['date']).tz_localize(None)
                    audio_url = msg['recording_url']

                    # Sync timing maps
                    closest_tele_idx = (tel_driver['Date'] - msg_time).abs().argsort()[:1]
                    if len(closest_tele_idx) == 0:
                        continue

                    idx = closest_tele_idx[0]
                    telemetry_slice = tel_driver.loc[max(0, idx - 100):idx]

                    features = extract_telemetry_features(telemetry_slice)
                    if not features:
                        continue

                    # Transcribe stream frames via temporary audio buffers
                    try:
                        audio_res = requests.get(audio_url, timeout=10)
                        if len(audio_res.content) < 1000:
                            continue

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                            temp_audio.write(audio_res.content)
                            temp_file_path = temp_audio.name

                        prompt = f"Formula 1 team radio. {driver} speaking to engineer. Terms: box, tyres, pit, engine, understeer."
                        result = model.transcribe(
                            temp_file_path, fp16=False, language="en",
                            temperature=0.0, initial_prompt=prompt
                        )
                        transcript = result['text'].strip()
                        os.remove(temp_file_path)

                        # Filter empty entries or audio static hallucinations
                        if "speaking to engineer" in transcript or not transcript or len(transcript) < 3:
                            continue

                        # Combine features with target strings
                        row_data = {
                            "year": target_year,
                            "location": location,
                            "driver": driver,
                            "timestamp": msg_time,
                            **features,
                            "transcript": transcript
                        }
                        collected_rows.append(row_data)

                    except Exception:
                        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        continue

                # Append data block instantly to protect against pipeline disconnect structural faults
                if collected_rows:
                    df_chunk = pd.DataFrame(collected_rows)
                    df_chunk.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                    print(f"    [Saved]: Appended {len(collected_rows)} training rows for {driver}.")

                time.sleep(0.5)  # Modest pacing delay to avoid hitting OpenF1 server rate ceilings

    print(f"\n🏁 Pipeline execution finished! Full multi-season database compiled in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()