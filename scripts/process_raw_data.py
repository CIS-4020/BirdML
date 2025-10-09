import cv2
import os
import shutil

firstClass = 295
boxDimension = 224

with open("../raw_data/classes.txt", "r") as file:
    with open("../processed_data/classes.txt", "w") as newFile:

        content = file.read()

        lines = content.strip().split("\n")

        for line in lines:
            classNum, className = line.split(" ", maxsplit=1)

            newClassNum = int(classNum) - firstClass

            if newClassNum > -1:
                print(className)
                newFile.write(str(newClassNum) + " " + className + "\n")
                
rawImgPath = "../raw_data/images"
newImgPath = "../processed_data/images"
if not os.path.exists(f"{newImgPath}"): os.mkdir(f"{newImgPath}")

for folder in os.listdir(rawImgPath):

    
    newFolder = str(int(folder)-int(firstClass))

    if os.path.exists(f"{newImgPath}/{newFolder}"):
        shutil.rmtree(f"{newImgPath}/{newFolder}")
    
    os.mkdir(f"{newImgPath}/{newFolder}")

    for file in os.listdir(f"{rawImgPath}/{folder}"):

        img = cv2.imread(f"{rawImgPath}/{folder}/{file}")

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

        cv2.imwrite(f"{newImgPath}/{newFolder}/{file}", squared_img)