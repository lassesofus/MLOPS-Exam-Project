FROM python:3.9-slim

EXPOSE $PORT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py setup.py
COPY src/ src/
COPY api/ api/

RUN pip install -e .
RUN pip install -r api/api_requirements.txt

WORKDIR /api/app

CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1
