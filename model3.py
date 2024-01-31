from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
import requests
import matplotlib.pyplot as plt

# dataset = load_dataset("huggingface/cats-image")
# image_path = dataset["test"]["image"][0]
# image = Image.open(image_path)

url = "https://d-art.ppstatic.pl/kadry/k/r/4b/e0/644912102461b_o_full.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-101")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")

inputs = feature_extractor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_label]

# Wizualizacja wyniku klasyfikacji
fig, ax = plt.subplots(1)
ax.imshow(image)
ax.set_title(f"Predicted Class: {predicted_class}")
plt.show()
