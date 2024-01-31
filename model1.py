from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Foteczka
url = "https://d-art.ppstatic.pl/kadry/k/r/4b/e0/644912102461b_o_full.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Określenie parametrów modelu i procesora
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Zostawianie wykryć z detekcją < 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Wizualizacja wyników
fig, ax = plt.subplots(1)
ax.imshow(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]

    # Rysowanie prostokąta wokół wykrytego obiektu
    rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # Dodanie etykiety z nazwą i precyzją detekcji
    ax.text(box[0], box[1], f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", color='red')

plt.show()
