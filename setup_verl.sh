#!/bin/bash
# Setup script for VeRL with proper dependency management

set -e

echo "=== Setting up VeRL environment ==="

# Phase 1: Core dependencies (already done)
echo "✓ PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"
echo "✓ CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Phase 2: Install VeRL requirements in batches to avoid conflicts
echo "Installing VeRL requirements in batches..."

# Batch 1: Core ML libraries (skip flash-attn for now)
echo "Batch 1: Core ML libraries..."
pip install transformers accelerate datasets evaluate tokenizers

# Batch 2: Serving and inference
echo "Batch 2: Serving libraries..."
pip install vllm sglang fastapi uvicorn

# Batch 3: Training libraries
echo "Batch 3: Training libraries..."
pip install deepspeed wandb

# Batch 4: Other utilities
echo "Batch 4: Utility libraries..."
pip install hydra-core omegaconf

# Phase 3: Install protobuf fixes
echo "Phase 3: Fixing protobuf versions..."
pip install --force-reinstall protobuf==5.29.5
pip install --force-reinstall --no-deps grpcio-status==1.71.0 selenium==4.33.0

# Phase 4: Install VeRL requirements without flash-attn
echo "Phase 4: Installing VeRL requirements (excluding flash-attn)..."
cd verl
cat requirements.txt | grep -v "flash_attn" | grep -v "^$" | xargs -n1 pip install --no-deps

echo "=== VeRL setup completed ==="
echo "Note: Flash Attention skipped due to missing CUDA toolkit"
echo "This is okay for inference, but may affect training performance"
