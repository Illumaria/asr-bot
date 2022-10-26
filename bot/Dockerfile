FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY bot.py bot.py

CMD ["python", "bot.py"]
