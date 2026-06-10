"""
Training + evaluation algorithm for the colorization model.

Kept separate from main.py so main.py is just config + orchestration.
"""

import os
import time
import json
import math
from datetime import datetime

import cv2
import numpy as np
import torch
from torch import nn


class ColorMSELoss(nn.Module):
    """
    MSE on the color channels only (skips channel 0).
    Useful if you ever switch the target to a LAB-style representation
    where channel 0 is luminance. With plain RGB it's equivalent to
    dropping the red channel, so for RGB targets just use nn.MSELoss.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        return self.mse(output[:, 1:], target[:, 1:])


def _save_sample(out_tensor, tgt_tensor, save_dir, epoch):
    # write a (pred, target) pair so we can eyeball progress
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_img = out_tensor.cpu().numpy().transpose(1, 2, 0)
    out_img = (out_img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"pred_{epoch:03d}_{stamp}.png"),
                cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    tgt_img = tgt_tensor.cpu().numpy().transpose(1, 2, 0)
    tgt_img = (tgt_img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"target_{epoch:03d}_{stamp}.png"),
                cv2.cvtColor(tgt_img, cv2.COLOR_RGB2BGR))


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running += loss.item()
    return running / max(1, len(loader))


def eval_one_epoch(model, loader, loss_fn, device, save_dir=None, epoch=None):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            running += loss_fn(outputs, targets).item()

            # only dump from the first batch of the eval pass
            if save_dir is not None and batch_idx == 0 and epoch is not None:
                _save_sample(outputs[0], targets[0], save_dir, epoch)
    return running / max(1, len(loader))


def fit(model, train_loader, test_loader, optimizer, loss_fn,
        epochs, device, save_dir, sample_every=10):
    """
    Full train/eval loop. Saves a checkpoint + sample images every
    `sample_every` epochs. Returns the per-epoch loss history.
    """
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        do_dump = (epoch % sample_every == 0)
        test_loss = eval_one_epoch(
            model, test_loader, loss_fn, device,
            save_dir=save_dir if do_dump else None,
            epoch=epoch if do_dump else None,
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        print(f"epoch {epoch:03d}  train {train_loss:.4f}  test {test_loss:.4f}")

        if do_dump:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"model_epoch_{epoch:03d}.pth"))

    return history


def benchmark(model, test_loader, device, save_path, history=None):
    """
    Run inference over the full test set, collect metrics, dump a JSON
    report at `save_path`. Returns the report dict.

    Metrics:
        - avg_mse           : mean squared error in [0,1] pixel space
        - avg_psnr_db       : peak signal-to-noise ratio (higher is better)
        - seconds_per_image : inference latency
        - images_per_second : throughput
    """
    model.eval()
    mse_fn = nn.MSELoss(reduction="mean")

    sum_mse = 0.0
    sum_psnr = 0.0
    n = 0
    elapsed = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            t0 = time.time()
            outputs = model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - t0

            batch_mse = mse_fn(outputs, targets).item()
            # PSNR for values in [0,1]: 10*log10(1/MSE)
            psnr = 10.0 * math.log10(1.0 / max(batch_mse, 1e-10))

            bs = inputs.size(0)
            sum_mse += batch_mse * bs
            sum_psnr += psnr * bs
            n += bs

    avg_mse = sum_mse / max(1, n)
    avg_psnr = sum_psnr / max(1, n)
    sec_per_img = elapsed / max(1, n)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "test_images": n,
        "avg_mse": round(avg_mse, 6),
        "avg_psnr_db": round(avg_psnr, 3),
        "seconds_per_image": round(sec_per_img, 6),
        "images_per_second": round(1.0 / max(sec_per_img, 1e-9), 2),
    }
    if history is not None and history.get("train_loss"):
        report["final_train_loss"] = round(history["train_loss"][-1], 6)
        report["final_test_loss"]  = round(history["test_loss"][-1], 6)
        report["epochs_trained"]   = len(history["train_loss"])

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- benchmark ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"saved to {save_path}")
    return report
