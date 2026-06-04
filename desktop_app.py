import customtkinter as ctk
import fastf1
import requests
import pandas as pd
import os
import whisper
import tempfile
import ssl
import joblib
import threading

# Setup security and caching
ssl._create_default_https_context = ssl._create_unverified_context
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

# Configure theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class F1AnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("F1 Telemetry & AI Stress Analyzer")
        self.geometry("900x700")

        # Initialize placeholders for the models
        self.whisper_model = None
        self.rf_model = None

        # --- UI LAYOUT ---
        # Left Control Panel
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_title = ctk.CTkLabel(self.sidebar, text="F1 AI CONFIG", font=ctk.CTkFont(size=20, weight="bold"))
        self.sidebar_title.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Season Choice
        self.label_year = ctk.CTkLabel(self.sidebar, text="Select Season:")
        self.label_year.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.combo_year = ctk.CTkComboBox(self.sidebar, values=["2023", "2024"], command=self.update_races)
        self.combo_year.grid(row=2, column=0, padx=20, pady=10)

        # Race Choice
        self.label_race = ctk.CTkLabel(self.sidebar, text="Select Grand Prix:")
        self.label_race.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.combo_race = ctk.CTkComboBox(self.sidebar, values=["Select a year first"], width=200)
        self.combo_race.grid(row=4, column=0, padx=20, pady=10)

        # Driver Entry
        self.label_driver = ctk.CTkLabel(self.sidebar, text="Driver Code (e.g., LEC):")
        self.label_driver.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_driver = ctk.CTkEntry(self.sidebar, placeholder_text="VER")
        self.entry_driver.grid(row=6, column=0, padx=20, pady=10)
        self.entry_driver.insert(0, "LEC")

        # Run Button
        self.btn_run = ctk.CTkButton(self.sidebar, text="Run AI Pipeline", command=self.start_analysis_thread,
                                     fg_color="#E10600", hover_color="#B30500")
        self.btn_run.grid(row=7, column=0, padx=20, pady=30)

        # Main Display Log Window
        self.display_box = ctk.CTkTextbox(self, width=600, font=ctk.CTkFont(family="Courier", size=12))
        self.display_box.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # --- THE FIX: Spawn thread ONLY after UI layout is 100% complete ---
        self.log("Initializing Dashboard... Loading machine learning arrays.")
        self.update_races("2023")  # Trigger baseline load

        # Now it is safe to boot up the models!
        threading.Thread(target=self.load_ai_models, daemon=True).start()

    def log(self, message):
        """Appends output to the central terminal view window safely."""
        self.display_box.insert("end", f"{message}\n")
        self.display_box.see("end")

    def load_ai_models(self):
        self.log("System: Downloading/Loading Whisper weights in background...")
        self.whisper_model = whisper.load_model("medium")
        self.log("System: Whisper AI online.")
        try:
            self.rf_model = joblib.load("f1_stress_model.pkl")
            self.log("System: Kinematic Stress Model (Random Forest) loaded successfully.")
        except FileNotFoundError:
            self.log("ERROR: f1_stress_model.pkl missing. Run train_model.py first!")

    def update_races(self, selected_year):
        url = f"https://api.openf1.org/v1/sessions?year={selected_year}&session_name=Race"
        try:
            self.races_raw = requests.get(url).json()
            names = [f"{r['country_name']} - {r['location']}" for r in self.races_raw]
            self.combo_race.configure(values=names)
            self.combo_race.set(names[0] if names else "")
        except Exception:
            self.log("Network Error: Could not connect to OpenF1 feeds.")

    def start_analysis_thread(self):
        """Spawns execution processing on an alternate thread to keep the window interactive."""
        threading.Thread(target=self.execute_analysis, daemon=True).start()

    def execute_analysis(self):
        if not self.whisper_model or not self.rf_model:
            self.log("Hold on! Models are still booting up in the background...")
            return

        year = int(self.combo_year.get())
        driver = self.entry_driver.get().strip().upper()
        race_idx = self.combo_race.get()

        selected_race = next(r for r in self.races_raw if f"{r['country_name']} - {r['location']}" == race_idx)
        session_key = selected_race['session_key']
        location = selected_race['location']

        self.log(f"\n⚡ Ingesting {year} {location} telemetry maps for {driver}...")

        try:
            session = fastf1.get_session(year, location, 'R')
            session.load(telemetry=True, weather=False)

            driver_num = session.get_driver(driver)['DriverNumber']
            tel_driver = session.laps.pick_drivers(driver).get_car_data()
        except Exception as e:
            self.log(f"Data Load Failure: Check driver layout parameters. Error: {e}")
            return

        self.log("-> Syncing live audio radio channels from OpenF1 server tower...")
        url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}&driver_number={driver_num}"
        radio_messages = requests.get(url).json()

        if isinstance(radio_messages, dict) or not radio_messages:
            self.log("No radio traffic data files discovered for this combination.")
            return

        self.log(f"-> Processing top transmission logs for {driver}:")

        for msg in radio_messages[:8]:
            radio_time = pd.to_datetime(msg['date']).tz_localize(None)
            audio_url = msg['recording_url']

            closest_tele = tel_driver.iloc[(tel_driver['Date'] - radio_time).abs().argsort()[:1]]
            if closest_tele.empty: continue

            idx = closest_tele.index[0]
            tele_slice = tel_driver.loc[max(0, idx - 100):idx]

            # Feature extraction for prediction array matrix
            features = pd.DataFrame([{
                "avg_speed": round(tele_slice['Speed'].mean(), 2),
                "max_speed": int(tele_slice['Speed'].max()),
                "speed_variance": round(tele_slice['Speed'].var(), 2) if not pd.isna(
                    tele_slice['Speed'].var()) else 0.0,
                "throttle_volatility": round(tele_slice['Throttle'].diff().abs().mean(), 2) if not pd.isna(
                    tele_slice['Throttle'].diff().abs().mean()) else 0.0,
                "throttle_snaps": int(
                    tele_slice['Throttle'].diff().abs()[tele_slice['Throttle'].diff().abs() > 20].count()),
                "brake_switches": int(tele_slice['Brake'].astype(int).diff().abs()[
                                          tele_slice['Brake'].astype(int).diff().abs() > 0].count()),
                "avg_rpm": round(tele_slice['RPM'].mean(), 1),
                "max_rpm": int(tele_slice['RPM'].max())
            }])

            stress_score = round(self.rf_model.predict(features)[0], 2)

            self.log("\n" + "=" * 45)
            self.log(f"TIMESTAMP: {radio_time.time()} | SPEED: {closest_tele['Speed'].values[0]} km/h")
            self.log(f"AI PREDICTED DRIVER STRESS LEVEL: {stress_score}/10.0")

            # Download & Transcribe
            try:
                audio_res = requests.get(audio_url)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_res.content)
                    tmp_path = tmp.name

                result = self.whisper_model.transcribe(tmp_path, fp16=False, language="en", temperature=0.0)
                transcript = result['text'].strip()
                os.remove(tmp_path)

                if "speaking to engineer" in transcript.lower() or not transcript:
                    self.log("TRANSCRIPT: [Static Engine noise only]")
                else:
                    self.log(f"TRANSCRIPT: \"{transcript}\"")
            except Exception:
                self.log("TRANSCRIPT: [Audio frames skipped/unreadable]")


if __name__ == "__main__":
    app = F1AnalyzerApp()
    app.mainloop()