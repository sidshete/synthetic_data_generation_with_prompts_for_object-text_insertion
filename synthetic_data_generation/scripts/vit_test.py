from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch



processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


image = Image.open("/home/dfki.uni-bremen.de/sshete/openai/smart_object_placement/data/source_images/000000013923.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)


target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]


for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.5:
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at {box.tolist()}")