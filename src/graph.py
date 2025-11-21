import matplotlib.pyplot as plt

#Expecting: epoch_data: list({"train_loss", "test_loss", "accuracy"})
def graphEpochData(epoch_data):
	x = []
	yTrain = []
	yTest = []
	yAcc = []
	for i in range(len(epoch_data)):
		x.append(i)
		yTrain.append(epoch_data[i]["train_loss"])
		yTest.append(epoch_data[i]["test_loss"])
		yAcc.append(epoch_data[i]["accuracy"])

	plt.plot(x, yTrain, label='Avg. Train Loss')
	plt.plot(x, yTest, label='Avg. Test Loss')
	plt.plot(x, yAcc, label='Avg. Accuracy')

	plt.xlabel("Epoch")
	plt.ylabel("Loss/Accuracy")
	plt.title("Average Loss + Accuracy per Train Epoch")
	plt.legend()

	plt.show()
