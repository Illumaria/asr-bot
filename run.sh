#!/usr/bin/env bash

echo "Starting Triton Inference Server container..."
TRITON_IMAGE_NAME=nvcr.io/nvidia/tritonserver:21.09-py3
docker pull $TRITON_IMAGE_NAME
docker run -d --rm --name triton --net bridge --add-host=host.docker.internal:host-gateway \
       -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/models:/models $TRITON_IMAGE_NAME \
       tritonserver --model-repository=/models --strict-model-config=false

echo "Starting backend container..."
docker build --rm -t asr-backend -f backend/Dockerfile backend
docker run -d --rm --name asr-backend --net bridge --add-host=host.docker.internal:host-gateway -e URL=host.docker.internal:8001 -p 5000:5000 asr-backend

echo "All done!"
