import os
import tempfile
import requests


def transcribe_audio_url(url: str, driver_code: str, model) -> str:
    """
    Download an MP3 from `url`, transcribe it locally with Whisper, and return the transcript.
    Returns an empty string if audio is empty, corrupt, or contains only engine noise.
    """
    try:
        response = requests.get(url, timeout=10)
        if len(response.content) < 1000:
            return ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        prompt = (
            f"Formula 1 team radio. {driver_code} speaking to race engineer. "
            "Terms: box, tyres, DRS, pit lane, braking, understeer, oversteer, apex, sector, lap time."
        )

        result = model.transcribe(
            tmp_path,
            fp16=False,
            language="en",
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            initial_prompt=prompt,
        )

        transcript = result["text"].strip()
        os.remove(tmp_path)

        # Suppress hallucinations where Whisper echoes back its own prompt
        if "speaking to race engineer" in transcript.lower() or len(transcript) < 3:
            return ""

        return transcript

    except Exception:
        if "tmp_path" in dir() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return ""
