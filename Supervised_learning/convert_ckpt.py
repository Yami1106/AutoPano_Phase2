#!/usr/bin/env python3
import torch
import os
from Network.Network import HomographyModel

# UPDATE THESE PATHS to match your setup
CKPT_PATH = "Phase2/Checkpoints/OneCycleLR_best_model_Train Loss: 4.8374 | Val Loss (Avg.): 5.5310 | Per-pixel loss (Avg.): 8.6642px .ckpt"
OUTPUT_PATH = "wrapper_compatible_best.pt"

print("Converting Lightning checkpoint...")
checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

# Extract hparams
hparams = checkpoint.get("hyper_parameters", {"InputSize": 128, "OutputSize": 8})
print(f"Using hparams: {hparams}")

# Create your model
model = HomographyModel(hparams)

# Load state_dict (handles Lightning 'model.' prefix automatically with strict=False)
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Save stitching-compatible format
torch.save({"model_state_dict": model.state_dict()}, OUTPUT_PATH)
print(f"Saved stitching-ready 'best.pt' to: {os.path.abspath(OUTPUT_PATH)}")
