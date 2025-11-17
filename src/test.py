import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.class_num_to_name import convertClassNumToClassName

import predict

testImgPath = "train_test_data/test/images"

if __name__ == "__main__":

	testImages = []

	if(len(sys.argv) < 2):
		print(f"Missing argument: Need number for choice of trained model")
		print("Example: python3 src/test.py 10")
		sys.exit(1)

	modelNum = sys.argv[1]
	modelName = f"birdML_{modelNum}_birds.pth"

	successRates = list()

	for imgFolder in range(int(modelNum)):
		imgCount = 0
		successCount = 0

		for imageName in os.listdir(f"{testImgPath}/{imgFolder}"):
			image = Image.open(f"{testImgPath}/{imgFolder}/{imageName}").convert("RGB")
			predictedClassInt = predict.predict(modelName, image, f"{convertClassNumToClassName(imgFolder)} Image#{imgCount}")
			if int(imgFolder) == predictedClassInt:
				successCount+=1
			imgCount+=1

		successRates.append(successCount/imgCount*100)
	print(successRates)