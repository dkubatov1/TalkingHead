#!/bin/bash

CheckpointsDir="./pretrained_models"

mkdir "pretrained_models"

pip install -U "huggingface_hub[hf_xet]"

# Mirror site to huggingface
export HF_ENDPOINT=https://hf-mirror.com

echo "Start download models from HuggingFace mirror site"

# Download Whisper weights (3.4 Gb)
hf download  TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir/musetalk-1.5 \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

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

# Download DWPose weights (407 Mb)
hf download yzd-v/DWPose \
    --local-dir $CheckpointsDir/dwpose \
    --include "dw-ll_ucoco_384.pth"

# Download Face Parse Bisent and ResNet weights (100 Mb)
hf download ManyOtherFunctions/face-parse-bisent \
    --local-dir $CheckpointsDir/face-parse-bisent \
    --include "79999_iter.pth" "resnet18-5c106cde.pth"

if [ $? -ne 0 ]; then
    echo "Failed to download models."
    exit 1
else
    echo "Download is successful."
fi
