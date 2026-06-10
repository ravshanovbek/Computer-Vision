import os
import random
from PIL import Image

categories = [
    "apparel", "artwork", "cars", "dishes",
    "illustrations", "landmark"
]

SRC = r"/Users/bekhzodravshanov/Desktop/PC/Sapienza/image_colorization/image dataset/128x128"
DST = "/Users/bekhzodravshanov/Desktop/PC/Sapienza/image_colorization/image dataset"
SEED = 42

# PNG was eating the disk (~50KB each x thousands x2 copies).
# JPEG q=90 is visually fine for 128x128 and ~10x smaller.
RGB_EXT = "jpg"
GREY_EXT = "jpg"
JPEG_Q = 90


def collect_images(src):
    items = []
    for c in categories:
        folder = os.path.join(src, c)
        if not os.path.isdir(folder):
            print("missing:", folder)
            continue
        for f in os.listdir(folder):
            ext = f.lower().rsplit(".", 1)[-1]
            if ext in ("jpg", "jpeg", "png", "bmp", "webp"):
                items.append((os.path.join(folder, f), c))
    return items


def make_dirs(dst):
    for mode in ["rgb", "greyscale"]:
        for split in ["train", "test"]:
            for c in categories:
                os.makedirs(os.path.join(dst, mode, split, c), exist_ok=True)


def convert_and_save(items, split, dst):
    for i, (path, cat) in enumerate(items):
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print("skip", path, "-", e)
            continue

        if im.size != (128, 128):
            im = im.resize((128, 128), Image.LANCZOS)

        grey = im.convert("L")

        base = f"{cat}_{i:06d}"
        rgb_path = os.path.join(dst, "rgb", split, cat, base + "." + RGB_EXT)
        grey_path = os.path.join(dst, "greyscale", split, cat, base + "." + GREY_EXT)

        try:
            im.save(rgb_path, quality=JPEG_Q, optimize=True)
            grey.save(grey_path, quality=JPEG_Q, optimize=True)
        except OSError as e:
            # most likely no space left - bail out instead of half-writing thousands more
            print("write failed:", e)
            raise

        if i % 500 == 0 and i > 0:
            print(f"  {split}: {i} done")


def process_dataset(src=SRC, dst=DST, train_ratio=0.8, seed=SEED):
    all_items = collect_images(src)
    if not all_items:
        raise RuntimeError("nothing found in " + src)

    print("found", len(all_items), "images across", len(categories), "categories")

    random.seed(seed)
    random.shuffle(all_items)

    cut = int(len(all_items) * train_ratio)
    train = all_items[:cut]
    test = all_items[cut:]

    make_dirs(dst)

    convert_and_save(train, "train", dst)
    convert_and_save(test, "test", dst)

    print("train:", len(train), "| test:", len(test))


if __name__ == "__main__":
    process_dataset()