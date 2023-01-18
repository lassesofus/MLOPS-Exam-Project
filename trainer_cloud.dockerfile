# Base image
FROM gcr.io/deeplearning-platform-release/pytorch-gpu


# Install python
# RUN apt update && \
# 	apt install --no-install-recommends -y build-essential gcc && \
# 	apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY reports/ reports/
COPY hydra_config/ hydra_config/

WORKDIR /
RUN pip install -e .
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_cloud.py"]