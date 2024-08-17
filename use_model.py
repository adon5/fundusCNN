import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('resnet50_fundus_trained.pth'))
model.eval()

output_classes = {0: "cataract", 1: "diabetic_retinopathy", 2: "glaucoma", 3: "normal"}

# This is the same transform used when training
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = ["./data/normal/8_left.jpg", "./data/cataract/103_left.jpg", "./data/normal/951_left.jpg"]
input_batch = []
for image_path in image_paths:
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch.append(input_tensor)

input_batch = torch.stack(input_batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)
model = model.to(device)

with torch.no_grad():
    outputs = model(input_batch)

probabilities = torch.nn.functional.softmax(outputs, dim=1)
predicted_classes = torch.argmax(probabilities, dim=1)

for i, image_path in enumerate(image_paths):
    print(f"Image: {image_path}, Predicted Class: {output_classes[int(predicted_classes[i])]}")