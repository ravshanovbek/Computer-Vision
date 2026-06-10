"""
Train the colorization model end-to-end.

Everything lives here except the Dataset class (in My_dataloader.py).
Run:  python main.py
"""

import os
import json
import time
import math
from datetime import datetime

import numpy as np
import cv2
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

from My_dataloader import MyDataLoader


# -------------------- config --------------------
DATA_ROOT = "/Users/bekhzodravshanov/Desktop/PC/Sapienza/image_colorization/image dataset"

PATH_TRAIN_GRAY = os.path.join(DATA_ROOT, "greyscale", "train")
PATH_TRAIN_RGB  = os.path.join(DATA_ROOT, "rgb",       "train")
PATH_TEST_GRAY  = os.path.join(DATA_ROOT, "greyscale", "test")
PATH_TEST_RGB   = os.path.join(DATA_ROOT, "rgb",       "test")

SAVE_DIR       = "saved"
METRICS_FILE   = "metrics.csv"
BENCHMARK_FILE = "benchmark.json"
PLOT_FILE      = "convergence.png"

# Model checkpoints go to a separate folder, every CHECKPOINT_EVERY epochs.
CHECKPOINT_DIR   = "/Users/bekhzodravshanov/Desktop/PC/Sapienza/image_colorization/Pretrained models"
CHECKPOINT_EVERY = 10

BATCH_SIZE   = 16
EPOCHS       = 50
LR           = 1e-3
SAMPLE_EVERY = 5    # save preview imgs every N epochs (separate from checkpoints)
NUM_WORKERS  = 0

# Use only this fraction of train/test for faster iteration.
# 1.0 = full dataset, 0.1 = 10%, etc. Same fraction applies to both splits.
DATA_FRACTION = 0.03
SUBSET_SEED   = 42
# ------------------------------------------------


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def maybe_subsample(dataset, fraction, seed, name=""):
    # grab a reproducible random fraction of the dataset
    if fraction >= 1.0:
        return dataset
    n_total = len(dataset)
    n_keep = max(1, int(n_total * fraction))
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=g)[:n_keep].tolist()
    print(f"  using {n_keep}/{n_total} {name} images ({fraction:.0%})")
    return torch.utils.data.Subset(dataset, indices)


# -------------------- model --------------------
# Upsample decoder - same shape as app.py / model.py so the saved weights
# can be dropped straight into the Gradio demo.
class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,   64,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),           nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64,  3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,  32,  3, padding=1), nn.ReLU(),
            nn.Conv2d(32,  16,  3, padding=1), nn.ReLU(),
            nn.Conv2d(16,  3,   3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# -------------------- helpers --------------------
def save_sample(out_t, tgt_t, save_dir, epoch):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred = (out_t.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    tgt  = (tgt_t.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"pred_{epoch:03d}_{stamp}.png"),
                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"target_{epoch:03d}_{stamp}.png"),
                cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR))


def append_metrics(path, row, header):
    new_file = not os.path.exists(path)
    with open(path, "a") as f:
        if new_file:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


def psnr_from_mse(mse, max_val=1.0):
    # PSNR for values in [0,1]: 10 * log10(1/MSE)
    return 10.0 * math.log10((max_val ** 2) / max(mse, 1e-10))


def plot_convergence(history, save_path):
    epochs = list(range(len(history["train_loss"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train", linewidth=2)
    ax1.plot(epochs, history["test_loss"],  label="test",  linewidth=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("MSE loss")
    ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["psnr"], color="tab:green", linewidth=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("Peak Signal-to-Noise Ratio")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Training convergence")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print("saved convergence plot to", save_path)


# -------------------- loops --------------------
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


def eval_one_epoch(model, loader, loss_fn, device, epoch, save_dir=None, do_dump=False):
    model.eval()
    running = 0.0
    sum_psnr = 0.0
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_loss = loss_fn(outputs, targets).item()
            running += batch_loss

            bs = inputs.size(0)
            sum_psnr += psnr_from_mse(batch_loss) * bs
            n += bs

            if do_dump and batch_idx == 0 and save_dir is not None:
                save_sample(outputs[0], targets[0], save_dir, epoch)

    return running / max(1, len(loader)), sum_psnr / max(1, n)


def benchmark(model, test_loader, device, save_path, history):
    model.eval()
    mse_fn = nn.MSELoss()

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
            psnr = psnr_from_mse(batch_mse)

            bs = inputs.size(0)
            sum_mse  += batch_mse * bs
            sum_psnr += psnr * bs
            n += bs

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "test_images": n,
        "avg_mse": round(sum_mse / max(1, n), 6),
        "avg_psnr_db": round(sum_psnr / max(1, n), 3),
        "seconds_per_image": round(elapsed / max(1, n), 6),
        "images_per_second": round(n / max(elapsed, 1e-9), 2),
        "final_train_loss": round(history["train_loss"][-1], 6),
        "final_test_loss":  round(history["test_loss"][-1],  6),
        "epochs_trained":   len(history["train_loss"]),
    }
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- benchmark ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("saved to", save_path)
    return report


# -------------------- main --------------------
def main():
    device = pick_device()
    print("device:", device)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    transform = nn.Sequential(
        torchvision.transforms.ConvertImageDtype(torch.float32),
    )

    train_ds = MyDataLoader(PATH_TRAIN_GRAY, PATH_TRAIN_RGB, transform=transform)
    test_ds  = MyDataLoader(PATH_TEST_GRAY,  PATH_TEST_RGB,  transform=transform)
    print(f"train images: {len(train_ds)} | test images: {len(test_ds)}")

    train_ds = maybe_subsample(train_ds, DATA_FRACTION, SUBSET_SEED,     name="train")
    test_ds  = maybe_subsample(test_ds,  DATA_FRACTION, SUBSET_SEED + 1, name="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    model = ColorizationModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    # fresh metrics log every run
    metrics_path = os.path.join(SAVE_DIR, METRICS_FILE)
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    history = {"train_loss": [], "test_loss": [], "psnr": []}

    print("starting training")
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        do_dump = (epoch % SAMPLE_EVERY == 0) or (epoch == EPOCHS - 1)
        test_loss, psnr = eval_one_epoch(
            model, test_loader, loss_fn, device, epoch,
            save_dir=SAVE_DIR, do_dump=do_dump,
        )
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["psnr"].append(psnr)

        # only print every SAMPLE_EVERY epochs (and the final one) to keep
        # the console quiet - full per-epoch numbers still go into metrics.csv
        if epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:03d}  "
                  f"train {train_loss:.4f}  test {test_loss:.4f}  "
                  f"psnr {psnr:.2f} dB  ({dt:.1f}s)")

        append_metrics(
            metrics_path,
            [epoch, f"{train_loss:.6f}", f"{test_loss:.6f}", f"{psnr:.4f}", f"{dt:.2f}"],
            header=["epoch", "train_loss", "test_loss", "psnr_db", "seconds"],
        )

        # checkpoint (every CHECKPOINT_EVERY epochs, plus the last one)
        if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS - 1:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

    # final weights - same filename app.py looks for
    final_path = os.path.join(CHECKPOINT_DIR, "final_pretrained_model.pth")
    torch.save(model.state_dict(), final_path)
    print("saved final weights to", final_path)

    # convergence plot
    plot_convergence(history, os.path.join(SAVE_DIR, PLOT_FILE))

    # benchmark on the test set
    benchmark(
        model=model,
        test_loader=test_loader,
        device=device,
        save_path=os.path.join(SAVE_DIR, BENCHMARK_FILE),
        history=history,
    )


if __name__ == "__main__":
    main()