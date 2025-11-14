import cv2
import os
import shutil

firstClass = 295
boxDimension = 224

rawImgPath = "raw_data/images"
newImgPath = "processed_data/images"
if not os.path.exists(f"{newImgPath}"): os.mkdir(f"{newImgPath}")

nextClass = 0

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

			for imgFile in os.listdir(f"{rawImgPath}/{folderName}"):

				img = cv2.imread(f"{rawImgPath}/{folderName}/{imgFile}")

				if img is None:
					continue

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

				cv2.imwrite(f"{newImgPath}/{nextClass}/{imgFile}", squared_img)

			nextClass+=1