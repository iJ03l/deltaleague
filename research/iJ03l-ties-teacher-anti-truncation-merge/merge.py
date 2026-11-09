import os
import sys
import torch
from safetensors.torch import load_file, save_file

def main():
    m1_path = "/home/x-5194/NearRadar/experiment/model/model.safetensors" # fb984a19 (Rank 1)
    m2_path = "/home/x-5194/NearRadar/models/ac61dc20/model.safetensors"  # ac61dc20 (QuestA low-truncation)
    out_path = "/home/x-5194/NearRadar/experiment/edited.safetensors"

    if not os.path.exists(m1_path) or not os.path.exists(m2_path):
        print(f"Waiting for models to be ready. m1: {os.path.exists(m1_path)}, m2: {os.path.exists(m2_path)}")
        sys.exit(1)

    print("Loading Expert 1 (Rank 1: fb984a19)...")
    w1 = load_file(m1_path)
    print("Loading Expert 2 (QuestA hinted GRPO: ac61dc20)...")
    w2 = load_file(m2_path)

    merged = {}
    alpha = 0.25 # 75% 9B-teacher reasoning, 25% QuestA anti-truncation discipline

    print(f"Applying Model Soup / Linear Mode Connectivity Merge (alpha={alpha})...")
    for k in w1.keys():
        t1 = w1[k]
        if k in w2:
            t2 = w2[k]
            # Linear interpolation in BF16/FP32
            t1_f = t1.float()
            t2_f = t2.float()
            m = (1.0 - alpha) * t1_f + alpha * t2_f
            merged[k] = m.to(t1.dtype)
        else:
            merged[k] = t1

    # 1. Calibrate Final RMSNorm
    norm_key = "model.language_model.norm.weight"
    if norm_key in merged:
        merged[norm_key] = merged[norm_key] * 1.040
        print(f"✓ Calibrated {norm_key} by 1.040")

    # 2. Retain Active Phase-Gate Push in Layer 23
    down_key = "model.language_model.layers.23.mlp.down_proj.weight"
    if down_key in merged:
        down = merged[down_key].clone()
        down[:, 0] = down[:, 0] * 1.20
        down[:, 1] = down[:, 1] * 1.20
        merged[down_key] = down
        print(f"✓ Applied x1.20 phase-gate push on Layer 23 MLP")

    print(f"Saving merged checkpoint to {out_path}...")
    save_file(merged, out_path)
    print(f"✓ Successfully saved {out_path} ({os.path.getsize(out_path) / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
