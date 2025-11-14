import sys, os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
#Ex. model = "bird_resnet50.pth" (searches in our models folder)
def predict(modelName, image, imageName):
    # Recreate the same model architecture
    model = models.resnet50(weights=None)

    # Change the classifier head if you had modified it during training
    num_classes = 3   # (whatever your actual number of bird species was)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(f"./models/{modelName}", map_location=torch.device("cpu")))
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

    # Commented out because we will recieve a image selected by the user from the frontend
    # image_path = "../common_eider_3.png"
    # image = Image.open(image_path).convert("RGB")

    # Apply transforms and add batch dimension
    input_tensor = transform(image).unsqueeze(0)  # shape [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        predicted_class = predicted.item()

    print(f"Prediction Results: {predicted_class}, {convertClassNumToClassName(predicted_class)}, {imageName}")

    return predicted_class

def convertClassNumToClassName(classNum):

    with open("./processed_data/classes.txt", "r") as classFile:

        for line in classFile:

            splitLine = line.split(" ")

            _classNum = splitLine[0]

            if int(_classNum) == int(classNum):
                return " ".join(splitLine[1:]).strip()
            
    return "Could not determine class name."

if __name__ == "__main__":

    testImages = []
    #test all in test_images
    if sys.argv[1] == "-a":
        for imgFile in os.listdir(f"./processed_data/test_images/"):
            testImages.append(imgFile)
    else:
        testImages.append(sys.argv[1])

    for imageName in testImages:
        imagePath = f"./processed_data/test_images/{imageName}"
        image = Image.open(imagePath).convert("RGB")
        predict("bird_resnet50.pth", image, imageName)