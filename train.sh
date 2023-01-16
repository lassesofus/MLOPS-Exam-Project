#!/bin/bash

BUCKET_NAME=bucket-train-bert
JOB_NAME=job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=europe-west1 \
    --master-image-uri=gcr.io/dtu-mlops-374314/trainer:latest \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-standard-8 \
    --master-accelerator=type=nvidia-tesla-k80,count=1 \