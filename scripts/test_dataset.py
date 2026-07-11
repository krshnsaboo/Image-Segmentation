from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.segmentation_dataset import SegmentationDataset


def main():
    project_root = Path(__file__).resolve().parent.parent

    image_dir = project_root / "data" / "images"
    mask_dir = project_root / "data" / "masks"

    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir
    )

    print("=" * 50)
    print(f"Dataset Size : {len(dataset)}")

    image, mask = dataset[0]

    print(f"Image Shape  : {image.shape}")
    print(f"Mask Shape   : {mask.shape}")
    print(f"Image dtype  : {image.dtype}")
    print(f"Mask dtype   : {mask.dtype}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    images, masks = next(iter(loader))

    print(f"Batch Images : {images.shape}")
    print(f"Batch Masks  : {masks.shape}")

    print("=" * 50)


if __name__ == "__main__":
    main()