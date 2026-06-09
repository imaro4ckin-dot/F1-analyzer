FROM python:3.11-slim

# ffmpeg is required by openai-whisper for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper medium model so the first request isn't slow
RUN python -c "import whisper; whisper.load_model('medium')"

# Download NLTK vader lexicon
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

COPY . .

EXPOSE 8050

# 1 worker: app loads a 1.5 GB Whisper model — multiple workers would multiply RAM
CMD ["gunicorn", "app.dashboard:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "1", \
     "--timeout", "120", \
     "--log-level", "info"]
