# Steps:
# Make new directory: /train_test_data
# Read from /processed_data and create two sub-directorys in train_test_data: /train and /test
# Within each of these create a /images directory that will contain a subset of the images of each bird from /processed_data
import sys
import random
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

processedImgPath = "processed_data/images"

trainImgPath = "train_test_data"
testImgPath = "train_test_data"
if not os.path.exists(f"{trainImgPath}"): os.mkdir(f"{trainImgPath}")
if not os.path.exists(f"{testImgPath}"): os.mkdir(f"{testImgPath}")

trainImgPath = "train_test_data/train"
testImgPath = "train_test_data/test"
if not os.path.exists(f"{trainImgPath}"): os.mkdir(f"{trainImgPath}")
if not os.path.exists(f"{testImgPath}"): os.mkdir(f"{testImgPath}")

trainImgPath = "train_test_data/train/images"
testImgPath = "train_test_data/test/images"
if not os.path.exists(f"{trainImgPath}"): os.mkdir(f"{trainImgPath}")
if not os.path.exists(f"{testImgPath}"): os.mkdir(f"{testImgPath}")

for imgFolder in os.listdir(f"{processedImgPath}"):
	files = os.listdir(os.path.join(processedImgPath, imgFolder))

	# 90%/10% Train/Test Split
	trainImgAmount = int(len(files) * 0.9)
	trainImages = set(random.sample(files, trainImgAmount))
	testImages = set(files) - trainImages

	if os.path.exists(f"{trainImgPath}/{imgFolder}"):
		shutil.rmtree(f"{trainImgPath}/{imgFolder}")
	os.mkdir(f"{trainImgPath}/{imgFolder}")

	if os.path.exists(f"{testImgPath}/{imgFolder}"):
		shutil.rmtree(f"{testImgPath}/{imgFolder}")
	os.mkdir(f"{testImgPath}/{imgFolder}")

	for img in trainImages:
		shutil.copyfile(
			os.path.join(processedImgPath, imgFolder, img), 
			os.path.join(trainImgPath, imgFolder, img)
		)
	for img in testImages:
		shutil.copyfile(
			os.path.join(processedImgPath, imgFolder, img), 
			os.path.join(testImgPath, imgFolder, img)
		)
