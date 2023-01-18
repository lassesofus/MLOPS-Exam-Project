# Base image
FROM  nvcr.io/nvidia/pytorch:22.07-py3

# Install python
RUN apt update && \
	apt install --no-install-recommends -y build-essential gcc && \
	apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY hydra_config/ hydra_config/

# Not sure if needed
COPY cloudbuild.yaml cloudbuild.yaml

#WORKDIR /
RUN pip install -e .
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
