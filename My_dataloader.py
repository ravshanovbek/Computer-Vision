import os
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset


class MyDataLoader(Dataset):
    def __init__(self, img_x_path, img_y_path, transform=None):
        # img_x_path -> /Users/bekhzodravshanov/Desktop/PC/Sapienza/image_colorization/image dataset/greyscale/train
        # img_y_path -> /Users/bekhzodravshanov/Desktop/PC/Sapienza/image_colorization/image dataset/rgb/train
        # the two trees mirror each other (same categories, same filenames)
        self.img_x = img_x_path
        self.img_y = img_y_path
        self.transform = transform

        # walk every category subfolder and collect relative paths
        self.files = []
        for cat in sorted(os.listdir(self.img_x)):
            cat_dir = os.path.join(self.img_x, cat)
            if not os.path.isdir(cat_dir):
                continue
            for f in os.listdir(cat_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.files.append(os.path.join(cat, f))

        if not self.files:
            raise RuntimeError(f"no images found under {self.img_x}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel = self.files[idx]
        # read greyscale as RGB so it comes out as 3 identical channels
        # -> matches the model's Conv2d(3, 64, ...) input
        image_x = read_image(os.path.join(self.img_x, rel), ImageReadMode.RGB)
        image_y = read_image(os.path.join(self.img_y, rel), ImageReadMode.RGB)

        if self.transform:
            image_x = self.transform(image_x)
            image_y = self.transform(image_y)
        return image_x, image_y
