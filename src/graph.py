import matplotlib.pyplot as plt

#Expecting: fold_data: list(list(avg_test_loss))
def graphFoldData(fold_data):
	x = []

	for i, fold in enumerate(fold_data):
		x = list(range(len(fold)))
		plt.plot(x, fold, label=f'Fold #{i+1}')

	plt.xlabel("Epoch")
	plt.ylabel("Average Test Loss")
	plt.title("Average Loss per Fold over Epochs")
	plt.legend()

	plt.show()
