import gradio as gr
import torch

from configs.config import (
    NUM_CLASSES,
    BEST_CHECKPOINT,
)

from src.models.deeplabv3 import build_model
from src.inference.predictor import Predictor
from src.training.checkpoints import load_checkpoint


# -------------------------------------------------------
# Load model once
# -------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = build_model(
    num_classes=NUM_CLASSES,
    pretrained=False,
)

load_checkpoint(
    path=BEST_CHECKPOINT,
    model=model,
    device=device,
)

predictor = Predictor(
    model=model,
    device=device,
)


# -------------------------------------------------------
# Gradio callback
# -------------------------------------------------------

def segment_image(image):

    results = predictor.predict(image)

    return (
        # results["original"],
        results["overlay"],
        results["colored_mask"],
        results["confidence"],
    )


# -------------------------------------------------------
# Interface
# -------------------------------------------------------

demo = gr.Interface(
    fn=segment_image,

    inputs=gr.Image(
        type="numpy",
        label="Input Image",
    ),

    outputs=[
        # gr.Image(label="Original"),
        gr.Image(label="Overlay"),
        gr.Image(label="Segmentation Mask"),
        gr.Image(label="Confidence Map"),
    ],

    title="Semantic Image Segmentation",

    description=(
        "DeepLabV3-ResNet101 trained on the MS COCO dataset "
        "for semantic image segmentation."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)