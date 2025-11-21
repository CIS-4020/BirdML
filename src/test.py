import sys
import os
from PIL import Image
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.class_num_to_name import convertClassNumToClassName

import predict

testImgPath = "train_test_data/test/images"

def test(modelNum, foldNum):
	successRates = list()
	accuracyList = list()
	worstBirdName = ""
	worstBirdAccuracy = 100
	bestBirdName = ""
	bestBirdAccuracy = 0

	totalSuccessCount = 0
	totalImageCount = 0


	modelName = f"birdML_{modelNum}_birds_{foldNum}.pth"

	for imgFolder in range(int(modelNum)):
		imgCount = 0
		successCount = 0

		for imageName in os.listdir(f"{testImgPath}/{imgFolder}"):
			image = Image.open(f"{testImgPath}/{imgFolder}/{imageName}").convert("RGB")
			predictedClassInt, _ = predict.predict(modelName, image, f"{convertClassNumToClassName(imgFolder)} Image#{imgCount}")
			if int(imgFolder) == predictedClassInt:
				successCount+=1
			imgCount+=1

		bird = convertClassNumToClassName(imgFolder)
		acc = successCount / imgCount * 100

		totalSuccessCount += successCount
		totalImageCount += imgCount

		if (acc < worstBirdAccuracy):
			worstBirdName = bird
			worstBirdAccuracy = acc
		
		if (acc > bestBirdAccuracy):
			bestBirdAccuracy = acc
			bestBirdName = bird

		successRates.append(successRates.append({
			"Bird": bird,
			"%": acc
		}))

		accuracyList.append(acc)


	print(successRates)
	print(f"Median: {statistics.median(accuracyList)}%. Mean: {totalSuccessCount / totalImageCount * 100}%")
	print(f"Best Bird: {bestBirdName} with an accuracy of {bestBirdAccuracy * 100}%")
	print(f"Worst Bird: {worstBirdName} with an accuracy of {worstBirdAccuracy * 100}%")
		


if __name__ == "__main__":

	if(len(sys.argv) < 3):
		print(f"Missing arguments: Need number for choice of trained model and fold number")
		print("Example: python3 src/test.py 10 3")
		sys.exit(1)

	modelNum = sys.argv[1]
	foldNum = sys.argv[2]

	test(modelNum, foldNum)