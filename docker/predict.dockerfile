# Base image
FROM python:3.9-slim

# Install python
RUN apt update && \
	apt install --no-install-recommends -y build-essential gcc && \
	apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.pyaQ
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY hydra_config/ hydra_config/

# Not sure if needed
COPY cloudbuild.yaml cloudbuild.yaml


WORKDIR /
RUN pip install -e .
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]
