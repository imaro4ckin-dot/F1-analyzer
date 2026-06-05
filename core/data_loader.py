import ssl
import os
import fastf1
import requests
import pandas as pd

# macOS SSL bypass for FastF1 / model weight downloads
ssl._create_default_https_context = ssl._create_unverified_context

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(_CACHE_DIR)


def fetch_races(year: int) -> list:
    """Return list of race session dicts from OpenF1 for a given year (2023+)."""
    url = f"https://api.openf1.org/v1/sessions?year={year}&session_name=Race"
    try:
        data = requests.get(url, timeout=15).json()
        if isinstance(data, dict) or not data:
            return []
        return data
    except Exception:
        return []


def load_session(year: int, location: str) -> fastf1.core.Session:
    """Load and return a FastF1 Race session with telemetry."""
    session = fastf1.get_session(year, location, "R")
    session.load(telemetry=True, weather=False)
    return session


def get_driver_map(session: fastf1.core.Session) -> dict:
    """Return {abbreviation: driver_number} for all drivers in a session."""
    driver_map = {}
    for num in session.drivers:
        try:
            info = session.get_driver(num)
            if info.get("Abbreviation"):
                driver_map[info["Abbreviation"]] = num
        except Exception:
            continue
    return driver_map


def get_telemetry(session: fastf1.core.Session, driver_code: str) -> pd.DataFrame:
    """Return car telemetry DataFrame for a driver (Date, Speed, Throttle, Brake, RPM, …)."""
    tel = session.laps.pick_drivers(driver_code).get_car_data()
    # Ensure Date column is timezone-naive for consistent timestamp arithmetic
    if hasattr(tel["Date"].dtype, "tz") and tel["Date"].dtype.tz is not None:
        tel["Date"] = tel["Date"].dt.tz_convert(None)
    return tel


def get_track_coords(session: fastf1.core.Session, driver_code: str) -> pd.DataFrame:
    """
    Return position data (X, Y, Date) for drawing the track outline and placing radio markers.
    Coordinates are in metres from an arbitrary circuit origin.
    """
    pos = session.laps.pick_drivers(driver_code).get_pos_data()
    if hasattr(pos["Date"].dtype, "tz") and pos["Date"].dtype.tz is not None:
        pos["Date"] = pos["Date"].dt.tz_convert(None)
    return pos[["Date", "X", "Y"]].dropna()


def get_stint_data(session, driver_code: str) -> "pd.DataFrame":
    """
    Return per-lap stint/tyre information for a driver.
    Columns: LapNumber (int), Compound (str), StintStart (datetime), StintEnd (datetime).
    Uses FastF1's LapStartDate (already tz-naive absolute datetime) directly.
    Returns empty DataFrame with those columns on any failure.
    """
    _empty = pd.DataFrame(columns=["LapNumber", "Compound", "StintStart", "StintEnd"])
    try:
        laps = session.laps.pick_drivers(driver_code).reset_index()
        if laps.empty:
            return _empty

        laps_sorted = laps.sort_values("LapNumber").reset_index(drop=True)

        # LapStartDate is already tz-naive absolute datetime in FastF1
        rows = []
        for i, row in laps_sorted.iterrows():
            try:
                lap_start = pd.Timestamp(row["LapStartDate"])
                # Lap end = next lap's start; for last lap use LapTime if available
                if i + 1 < len(laps_sorted):
                    lap_end = pd.Timestamp(laps_sorted.loc[i + 1, "LapStartDate"])
                else:
                    lap_time = row.get("LapTime")
                    if pd.notna(lap_time):
                        lap_end = lap_start + pd.to_timedelta(lap_time)
                    else:
                        lap_end = lap_start + pd.Timedelta(seconds=120)  # fallback

                compound = str(row.get("Compound", "UNKNOWN") or "UNKNOWN").upper()
                rows.append({
                    "LapNumber": int(row["LapNumber"]),
                    "Compound": compound,
                    "StintStart": lap_start,
                    "StintEnd": lap_end,
                })
            except Exception:
                continue

        if not rows:
            return _empty
        return pd.DataFrame(rows)
    except Exception:
        return _empty


def fetch_race_control(session_key: int) -> list:
    """
    Return SC/VSC race control events from OpenF1 for a session.
    Each dict has: date (str ISO), category (str), message (str), flag (str).
    Returns [] on any failure or if no SC/VSC events found.
    """
    url = f"https://api.openf1.org/v1/race_control?session_key={session_key}"
    try:
        data = requests.get(url, timeout=10).json()
        if not isinstance(data, list):
            return []
        sc_keywords = {"safetycar", "vsc", "safety_car", "virtual_safety_car", "virtual safety car", "safety car"}
        result = []
        for item in data:
            category = str(item.get("category", "")).lower()
            flag = str(item.get("flag", "")).lower()
            message = str(item.get("message", "")).lower()
            if (category in sc_keywords
                    or flag in sc_keywords
                    or any(k in message for k in ("safety car", "virtual safety car", "safety car deployed",
                                                   "safety car in this lap"))):
                result.append({
                    "date": item.get("date", ""),
                    "category": item.get("category", ""),
                    "message": item.get("message", ""),
                    "flag": item.get("flag", ""),
                })
        return result
    except Exception:
        return []


def get_all_driver_codes(session) -> list:
    """Return sorted list of 3-letter driver abbreviations present in a session."""
    codes = []
    for num in session.drivers:
        try:
            info = session.get_driver(num)
            abbr = info.get("Abbreviation")
            if abbr:
                codes.append(abbr)
        except Exception:
            continue
    return sorted(set(codes))


def get_lap_times_data(session, driver_code: str) -> list:
    """
    Return per-lap timing data for the lap-time evolution chart.
    Each dict: {lap, lap_time_s, compound, is_sc, is_vsc}.
    Returns [] on any failure.
    """
    try:
        laps = session.laps.pick_drivers(driver_code).sort_values("LapNumber")
        result = []
        for _, row in laps.iterrows():
            lt = row.get("LapTime")
            if pd.isna(lt):
                continue
            lap_s = lt.total_seconds()
            if lap_s <= 0 or lap_s > 600:
                continue
            ts = str(row.get("TrackStatus", "1") or "1")
            result.append({
                "lap": int(row["LapNumber"]),
                "lap_time_s": round(lap_s, 3),
                "compound": str(row.get("Compound", "UNKNOWN") or "UNKNOWN").upper(),
                "is_sc":  "4" in ts,
                "is_vsc": "6" in ts,
            })
        return result
    except Exception:
        return []


def get_pit_stops(session, driver_code: str) -> list:
    """
    Return list of pit-in absolute timestamps (tz-naive) for a driver.
    Each dict: {lap (int), time (ISO string)}.
    Uses PitInTime (timedelta offset from session start) + session.t0_date.
    Returns [] on any failure.
    """
    try:
        t0 = session.t0_date
        if t0 is None:
            return []
        if hasattr(t0, "tz") and t0.tz is not None:
            t0 = t0.tz_convert(None)
        laps = session.laps.pick_drivers(driver_code)
        pit_laps = laps[laps["PitInTime"].notna()].sort_values("LapNumber")
        return [
            {"lap": int(row["LapNumber"]), "time": (t0 + row["PitInTime"]).isoformat()}
            for _, row in pit_laps.iterrows()
        ]
    except Exception:
        return []


def fetch_radio(session_key: int, driver_num) -> list:
    """
    Return list of team radio message dicts from OpenF1.
    Each dict has keys: date, recording_url.
    Returns empty list if no radio data is available.
    """
    url = f"https://api.openf1.org/v1/team_radio?session_key={session_key}&driver_number={driver_num}"
    try:
        data = requests.get(url, timeout=10).json()
        if isinstance(data, dict) or not data:
            return []
        return data
    except Exception:
        return []
