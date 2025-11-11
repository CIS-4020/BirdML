import torch
from torchvision import models

# Recreate the same model architecture
model = models.resnet50(weights=None)

# Change the classifier head if you had modified it during training
num_classes = 3   # (whatever your actual number of bird species was)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("bird_resnet50.pth", map_location=torch.device("cpu")))
model.eval()  # set to evaluation mode

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = "../common_eider_3.png"
image = Image.open(image_path).convert("RGB")

# Apply transforms and add batch dimension
input_tensor = transform(image).unsqueeze(0)  # shape [1, 3, 224, 224]

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    predicted_class = predicted.item()

print(predicted_class)