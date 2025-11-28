import sys
import os
from PIL import Image
import statistics
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.class_num_to_name import convertClassNumToClassName

import predict

testImgPath = "train_test_data/test/images"

def create_boxplot(accuracyList, modelNum):

	fig, ax = plt.subplots(figsize=(14, 6))

	bp = ax.boxplot(accuracyList, patch_artist=True)

	x_positions = np.random.normal(1, 0.04, size=len(accuracyList))
	
	ax.scatter(x_positions, accuracyList, alpha=1, color='black', zorder=3)
	
	# Customize colors
	for patch in bp['boxes']:
		patch.set_facecolor('lightblue')
		patch.set_alpha(0.7)
		patch.set_zorder(2)
	
	# Customize plot
	ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
	ax.set_title(f'Accuracy Distribution \n(Model: {modelNum} birds)', fontsize=14, fontweight='bold')
	ax.set_ylim(min(accuracyList)-10, min(max(accuracyList) + 10, 105))
	ax.grid(True, alpha=0.3, axis='y')
	
	plt.tight_layout()
	
	# Save plot
	output_path = f"results/boxplot_accuracy_{modelNum}_birds.png"
	os.makedirs("results", exist_ok=True)
	plt.savefig(output_path, dpi=100, bbox_inches='tight')
	print(f"Box-and-whisker plot saved to {output_path}")
	
	plt.show()

def test(modelNum):
	successRates = list()
	accuracyList = list()
	worstBirdName = ""
	worstBirdAccuracy = 100
	bestBirdName = ""
	bestBirdAccuracy = 0

	totalSuccessCount = 0
	totalImageCount = 0


	modelName = f"birdML_{modelNum}_birds.pth"

	for imgFolder in range(int(modelNum)):
		imgCount = 0
		successCount = 0
		birdAccuracies = []  # Store per-image results for this bird

		for imageName in os.listdir(f"{testImgPath}/{imgFolder}"):
			image = Image.open(f"{testImgPath}/{imgFolder}/{imageName}").convert("RGB")
			predictedClassInt, _ = predict.predict(modelName, image, f"{convertClassNumToClassName(imgFolder)} Image#{imgCount}")
			if int(imgFolder) == predictedClassInt:
				successCount+=1
				birdAccuracies.append(100)
			else:
				birdAccuracies.append(0)
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

		successRates.append({
			"Bird": bird,
			"%": acc
		})

		accuracyList.append(acc)

	for succ in successRates:
		print(f"{succ['Bird']}: {succ['%']}%")
	print(f"Median: {statistics.median(accuracyList)}%. Mean: {totalSuccessCount / totalImageCount * 100}%")
	print(f"Best Bird: {bestBirdName} with an accuracy of {bestBirdAccuracy}%")
	print(f"Worst Bird: {worstBirdName} with an accuracy of {worstBirdAccuracy}%")
	
	create_boxplot(accuracyList, modelNum)

if __name__ == "__main__":

	if(len(sys.argv) < 2):
		print(f"Missing argument: Need number for choice of trained model")
		print("Example: python3 src/test.py 10")
		sys.exit(1)

	modelNum = sys.argv[1]

	test(modelNum)