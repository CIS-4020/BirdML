import sys
import os
from PIL import Image
import torch

# pip install ultralytics
from ultralytics import YOLO
import cv2


def findAndCropBird(imagePath):

    croppedImagePath = "./processed_data/cropped_test_images"
    if not os.path.exists(f"{croppedImagePath}"): os.mkdir(f"{croppedImagePath}")
    
    yolo_model = YOLO("yolov8n.pt")

    image = cv2.imread(f"./processed_data/test_images/{imagePath}")
    if image is None:
        raise FileNotFoundError(f"Could not open image {imagePath}")

    # YOLO expects RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run detection
    results = yolo_model.predict(source=img_rgb, conf=0.3, classes=[14], verbose=False, show=False)  # class 14 = bird in COCO

    if len(results[0].boxes) == 0:
        print(f"No birds detected in {imagePath}. Using full image.")
        final_img = Image.fromarray(img_rgb)
    else:
            # Take first detected bird
            box = results[0].boxes.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box)
            cropped_img = img_rgb[y1:y2, x1:x2]

    final_img = Image.fromarray(resizeAndPaddBlack(cropped_img))

    final_img.save(f"{croppedImagePath}/cropped_{imagePath}")

    return final_img

def resizeAndPaddBlack(img):

    boxDimension = 224

    baseHeight, baseWidth, baseChannels = img.shape

    if baseWidth > baseHeight:
        ratio = baseHeight/baseWidth
        width = boxDimension
        height = int(boxDimension * ratio)
    else:
        ratio = baseWidth/baseHeight
        height = boxDimension
        width = int(boxDimension * ratio)

    resized_img = cv2.resize(img, (width, height))

    top = (boxDimension - height) // 2
    bottom = boxDimension - height - top
    left = (boxDimension - width) // 2
    right = boxDimension - width - left

    squared_img = cv2.copyMakeBorder(
        resized_img,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black padding
    )

    return squared_img

if __name__ == "__main__":
    imagePath = sys.argv[1]
    findAndCropBird(imagePath)

    

