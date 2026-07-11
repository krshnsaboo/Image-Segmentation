from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.config import IMAGE_SIZE


class Predictor:
    """
    Performs inference using a trained semantic segmentation model.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

    def predict(self, image):

        """
        Predict segmentation mask for an image.

        Args:
            image:
                PIL.Image
                OR
                image path (str / Path)

        Returns:
            Dictionary containing prediction results.
        """

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        original = image.resize(IMAGE_SIZE)

        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():

            output = self.model(input_tensor)["out"]

            probabilities = torch.softmax(output, dim=1)

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).squeeze(0).cpu().numpy()

            confidence = torch.max(
                probabilities,
                dim=1
            )[0].squeeze(0).cpu().numpy()

        color_mask = self._colorize_mask(prediction)

        overlay = (
            0.6 * np.array(original)
            + 0.4 * color_mask
        ).astype(np.uint8)

        confidence_map = (confidence * 255).astype(np.uint8)

        return {
            "original": np.array(original),
            "predicted_mask": prediction,
            "colored_mask": color_mask,
            "overlay": overlay,
            "confidence": confidence_map,
        }

    def _colorize_mask(self, mask):

        color_mask = np.zeros(
            (*mask.shape, 3),
            dtype=np.uint8
        )

        for cls in np.unique(mask):
            color_mask[mask == cls] = self.colors[cls]

        return color_mask