import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch


MODEL_NAME = "prithivMLmods/Fashion-Product-baseColour"


def _load_model_and_processor(model_name: str = MODEL_NAME):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiglipForImageClassification.from_pretrained(model_name)
    model.to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor, device


model, processor, device = _load_model_and_processor()


# Mapping taken from the model card of Fashion-Product-baseColour
ID2LABEL = {
    0: "Beige", 1: "Black", 2: "Blue", 3: "Bronze", 4: "Brown", 5: "Burgundy",
    6: "Charcoal", 7: "Coffee Brown", 8: "Copper", 9: "Cream", 10: "Fluorescent Green",
    11: "Gold", 12: "Green", 13: "Grey", 14: "Grey Melange", 15: "Khaki", 16: "Lavender",
    17: "Lime Green", 18: "Magenta", 19: "Maroon", 20: "Mauve", 21: "Metallic",
    22: "Multi", 23: "Mushroom Brown", 24: "Mustard", 25: "Navy Blue", 26: "Nude",
    27: "Off White", 28: "Olive", 29: "Orange", 30: "Peach", 31: "Pink", 32: "Purple",
    33: "Red", 34: "Rose", 35: "Rust", 36: "Sea Green", 37: "Silver", 38: "Skin",
    39: "Steel", 40: "Tan", 41: "Taupe", 42: "Teal", 43: "Turquoise Blue", 44: "White", 45: "Yellow"
}


def classify_base_color(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {ID2LABEL[i]: round(probs[i], 3) for i in range(len(probs))}
    return predictions


iface = gr.Interface(
    fn=classify_base_color,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Base Colour Prediction Scores"),
    title="Fashion-Product-baseColour",
    description=(
        "Upload a fashion product image to detect its primary color "
        "(e.g., Red, Black, Cream, Navy Blue, etc.)."
    ),
)


if __name__ == "__main__":
    iface.launch()


