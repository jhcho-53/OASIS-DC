import os
import torch

def save_ckpt(path, net, optimizer, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": meta,
    }, path)
    print(f"[ckpt] saved: {path}")