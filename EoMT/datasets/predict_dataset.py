from pathlib import Path
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import Dataset


class PredictDataset(Dataset):

    def __init__(self, image_dir, save_dir):
        self.image_dir = Path(image_dir)
        self.save_dir = save_dir

        self.files = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
            ]
        ])

        self.transform = T.PILToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return img, path.name, self.save_dir
