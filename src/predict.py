import sys, os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from scripts import class_num_to_name as classConvert

import findBird

def load_model(model_name):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "models", model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    return model_path

def predict(modelName, image, imageName):
    # Recreate the same model architecture
    model = models.resnet50(weights=None)

    num_classes = int(modelName.split("_")[1])
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model_path = load_model(modelName)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # set to evaluation mode

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0)  # shape [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        predicted_class = predicted.item()

    print(f"Prediction Results: {imageName} --> {classConvert.convertClassNumToClassName(predicted_class)}")

    return predicted_class

if __name__ == "__main__":

    testImages = []

    if(len(sys.argv) < 3):
        print(f"Missing argument: {len(sys.argv)}/3 arguments provided.")
        print("Example: python3 src/predict.py -a 10")
        sys.exit(1)

    #first arg is the image/images, use -a for all
    if sys.argv[1] == "-a":
        for imgFile in os.listdir(f"./test_images/"):
            testImages.append(imgFile)
    else:
        testImages.append(sys.argv[1])

    #second arg is for the model num
    modelNum = sys.argv[2]
    model = f"birdML_{modelNum}_birds.pth"

    for imageName in testImages:
        
        imagePath = f"./test_images/{imageName}"
        image = findBird.findAndCropBird(imageName)
        predict(model, image, imageName)