import sys
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torch import nn, optim
import torch

def trainModel(numFolders=-1):
	# Standard transform pipeline for ResNet input
	train_transforms = transforms.Compose([
		transforms.Resize((224, 224)),			   # ResNet expects 224x224
		transforms.ToTensor(),					   # Convert PIL -> Tensor
		transforms.Normalize(						# Normalize same as ImageNet
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	train_data = datasets.ImageFolder("processed_data/images", transform=train_transforms)

	if numFolders > -1:
		# Selecting a subset of the total bird classes
		all_classes = train_data.classes
		keep_classes = all_classes[:numFolders]
		keep_class_idxs = [train_data.class_to_idx[c] for c in keep_classes]
		subset_indices = [i for i, (_, label) in enumerate(train_data.samples)
						if label in keep_class_idxs]
		train_subset = Subset(train_data, subset_indices)

		train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
	else:
		train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

	# Use GPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

	# Replace final fully connected layer with correct output size
	num_classes = len(train_data.classes)
	model.fc = nn.Linear(model.fc.in_features, num_classes)
	model = model.to(device)

	# Define optimizer and loss function
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0003)

	# AI.TRAIN()
	for epoch in range(5):  # increase epochs for better training
		model.train()
		total_loss = 0
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")
		
	torch.save(model.state_dict(), "models/bird_resnet50.pth")

if __name__ == "__main__":
	if len(sys.argv) > 1:
		trainModel(int(sys.argv[1]))
	else:
		trainModel()
