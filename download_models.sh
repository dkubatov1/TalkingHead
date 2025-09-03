#!/bin/bash

CheckpointsDir="./pretrained_models"

mkdir "pretrained_models"

pip install -U "huggingface_hub[hf_xet]"

echo "Start download models from HuggingFace mirror site"

# Download Whisper weights (151 Mb)
hf download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download SD VAE weights (335 Mb)
hf download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download SyncNet weights (5.7 Gb)
hf download ByteDance/LatentSync-1.6 \
  --local-dir $CheckpointsDir/syncnet \
  --include "latentsync_syncnet.pt" "latentsync_unet.pt"

if [ $? -ne 0 ]; then
    echo "Failed to download models."
    exit 1
else
    echo "Download is successful."
fi
