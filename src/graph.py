import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    # Save plot
	output_path = f"results/foldData.png"
	os.makedirs("results", exist_ok=True)
	plt.savefig(output_path, dpi=100, bbox_inches='tight')
	print(f"Plot saved to {output_path}")

	plt.show()
