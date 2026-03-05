# Optional: Build a custom worker-vllm image with the Qwen model baked in.
# This eliminates the ~2-3 min model download on each worker cold start.
#
# Requires cloning the worker-vllm repo first:
#   git clone https://github.com/runpod-workers/worker-vllm.git
#   cp Dockerfile worker-vllm/Dockerfile
#   cd worker-vllm
#   docker build -t YOUR_DOCKER_USERNAME/qwen-story-server:1.0 \
#       --build-arg MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
#       --build-arg BASE_PATH="/models" \
#       --build-arg QUANTIZATION="bitsandbytes" .
#   docker push YOUR_DOCKER_USERNAME/qwen-story-server:1.0
#
# For most users, the pre-built image (runpod/worker-v1-vllm:stable-cuda12.1.0)
# with environment variables is simpler. See README.md for details.
#
# NOTE: This file is only used if you want to bake the model into the image.
# The primary deployment method uses RunPod's pre-built worker-vllm image
# configured entirely through environment variables in the RunPod console.
