import sys
import cv2
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import findBird

rawImgPath = "raw_data/images"
newImgPath = "processed_data/images"
singleImgPath = "single_data"
if not os.path.exists(f"{newImgPath}"): os.mkdir(f"{newImgPath}")
if not os.path.exists(f"{singleImgPath}"): os.mkdir(f"{singleImgPath}")

nextClass = 0
boxDimension = 224

# overhead for bounding boxes
bbox_dict = {}
with open("raw_data/bounding_boxes.txt", "r") as f:
	for line in f:
		parts = line.strip().split()
		img_name = parts[0].replace("-", "") + ".jpg"
		bbox = list(map(int, parts[1:])) # [x, y, width, height]
		bbox_dict[img_name] = bbox

open("processed_data/classes.txt", "w").close()

with open("raw_data/classes.txt", "r") as rawClassFile:

	content = rawClassFile.read()
	lines = content.strip().split("\n")

	with open("processed_data/classes.txt", "a") as newClassFile:

		for i in range(295, 1011):

			folderName = str(i)
			if i < 1000:
				folderName = "0" + folderName

			if not os.path.exists(f"{rawImgPath}/{folderName}"):
				continue

			if os.path.exists(f"{newImgPath}/{nextClass}"):
				shutil.rmtree(f"{newImgPath}/{nextClass}")
			
			os.mkdir(f"{newImgPath}/{nextClass}")

			classNum, className = lines[int(folderName)].split(" ", maxsplit=1)
			newClassFile.write(str(nextClass) + " " + className + "\n")
			newClassFile.flush()

			firstImage = True
			for imgFile in os.listdir(f"{rawImgPath}/{folderName}"):

				imageName = imgFile.split(".")[0]

				img = cv2.imread(f"{rawImgPath}/{folderName}/{imgFile}")

				if img is None:
					continue

				if firstImage:
					cv2.imwrite(f"{singleImgPath}/{nextClass}.jpg", img)

				#crop using bounding boxes
				if imgFile in bbox_dict:
					x, y, w, h = bbox_dict[imgFile]
					# Make sure bbox doesn't go outside image
					x1 = max(0, x)
					y1 = max(0, y)
					x2 = min(img.shape[1], x + w)
					y2 = min(img.shape[0], y + h)
					full_img = img[y1:y2, x1:x2]

				squared_img =  findBird.resizeAndPaddBlack(full_img)

				cv2.imwrite(f"{newImgPath}/{nextClass}/{imgFile}", squared_img)

				firstImage = False

			nextClass+=1