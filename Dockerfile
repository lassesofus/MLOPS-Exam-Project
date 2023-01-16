# Base image
FROM python:3.9-slim

# Install python
RUN apt update && \
	apt install --no-install-recommends -y build-essential gcc && \
	apt clean && rm -rf /var/lib/apt/lists/*

# Copy over our application (the essential parts) from our computer to the container
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY reports/ reports/
COPY cloudbuild.yaml cloudbuild.yaml

WORKDIR /
RUN pip install -e .
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
