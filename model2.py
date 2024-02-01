from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_object_detection_model2(file_path):
    # Check if the file_path is a URL
    if file_path.startswith('http://') or file_path.startswith('https://'):
        # If it's a URL, use requests to get the image
        response = requests.get(file_path, stream=True)
        response.raise_for_status()  # Check for errors
        image = Image.open(response.raw)
    else:
        # If it's a local file, open it directly
        image = Image.open(file_path)

    # Define model and processor
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    # Process the image and make predictions
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Keep detections with confidence > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # Visualization
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]

        # Draw rectangle around the detected object
        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add label with name and detection precision
        ax.text(box[0], box[1], f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", color='red')

    plt.show()