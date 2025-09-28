import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Fashion-Product-Season"  # Replace with your actual model path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    0: "Fall",
    1: "Spring",
    2: "Summer",
    3: "Winter"
}

def classify_season(image):
    """Predicts the most suitable season for a fashion product."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {id2label[i]: round(probs[i], 3) for i in range(len(probs))}
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_season,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Season Prediction Scores"),
    title="Fashion-Product-Season",
    description="Upload a fashion product image to predict its most suitable season (Fall, Spring, Summer, Winter)."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
