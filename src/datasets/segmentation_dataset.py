from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic image segmentation.

    Expected directory structure:
        images/
            000001.jpg
            000002.jpg
            ...
        masks/
            000001.png
            000002.png
            ...
    """

    def __init__(self, image_dir, mask_dir, image_transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(self.image_dir.glob("*"))
        self.masks = sorted(self.mask_dir.glob("*"))

        if len(self.images) != len(self.masks):
            raise RuntimeError(
                "Number of images and masks does not match."
            )

        self.max_class = 0

        for mask_path in self.masks[:1000]:
            mask = np.array(Image.open(mask_path))
            self.max_class = max(self.max_class, mask.max())

        self.num_classes = self.max_class + 1

        print(f"Loaded {len(self.images)} samples")
        print(f"Detected {self.num_classes} classes")
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in: {self.image_dir}")

        self.image_transform = image_transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        mask_path = self.mask_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image: {image_path.name}"
            )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.image_transform(image)
        mask = torch.from_numpy(np.array(mask)).long()
        if mask.max() >= self.num_classes:
            raise ValueError(
                f"Invalid class id {mask.max()} in {mask_path.name}"
            )
        return image, mask