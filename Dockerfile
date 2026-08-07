FROM python:3.12-slim

# ffmpeg is a hard dependency of the assembly stage, not an optional extra.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# When the running build is stale, every symptom is a lie: a fixed bug looks
# unfixed, and a config change looks ignored. Stamping the image gives the app
# something to show. It sits after COPY deliberately -- any source change busts
# the layer, so the stamp cannot outlive the code it describes.
RUN date -u +%Y-%m-%dT%H:%M:%SZ > /app/BUILD_STAMP

# The library lives on a mounted volume so it survives redeploys.
ENV PAPERPOD_DATA_DIR=/data \
    PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "app.py"]
