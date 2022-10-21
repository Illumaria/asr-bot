FROM python:3.10-slim

RUN apt-get update \
    && apt-get -y install libsndfile-dev libopus0 libopusfile0 libvorbisfile3

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt


COPY src src
COPY app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
