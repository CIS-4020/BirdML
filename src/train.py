import sys
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import ResNet50_Weights
from torch import nn, optim
import torch
from sklearn.model_selection import KFold
from graph import graphFoldData

# K-Fold Cross-Validation Train Function
def kFoldEval(numFolders=-1, num_epochs=20):

	# Random image tranformations to increase sample variety
	train_transforms = transforms.Compose([
		transforms.RandomHorizontalFlip(p=0.5),
    	transforms.RandomRotation(degrees=20),
		transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
		transforms.RandomGrayscale(p=0.1),
		transforms.ColorJitter(
			brightness=0.2,
			contrast=0.2,
			saturation=0.2,
			hue=0.1
		),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	# Three hyperparameter configurations for our k-fold cross-validation to try
	hyperparameter_configs = [
		{"lr": 1e-3, "weight_decay": 1e-4},
		{"lr": 5e-4, "weight_decay": 5e-5},
		{"lr": 1e-4, "weight_decay": 1e-5},
	]

	kf = KFold(n_splits=5, shuffle=True, random_state=42)

	train_data = datasets.ImageFolder("./train_test_data/train/images", transform=train_transforms)

	numFolders = min(numFolders, len(train_data.classes))
	if numFolders > -1:
		# Selecting a subset of the total bird classes. We allow this so that we can test training smaller models that don't take nearly as long to train.
		all_classes = sorted(train_data.classes, key=lambda x: int(x))
		keep_classes = all_classes[:numFolders]

		keep_class_idxs = [train_data.class_to_idx[c] for c in keep_classes]
		subset_indices = [i for i, (_, label) in enumerate(train_data.samples)
						if label in keep_class_idxs]
		
		label_map = {old: new for new, old in enumerate(keep_class_idxs)}

		class SubsetWithRemap(torch.utils.data.Dataset):
			def __init__(self, dataset, indices, label_map):
				self.dataset = dataset
				self.indices = indices
				self.label_map = label_map

			def __len__(self):
				return len(self.indices)

			def __getitem__(self, idx):
				image, label = self.dataset[self.indices[idx]]
				return image, self.label_map[label]

		train_subset = SubsetWithRemap(train_data, subset_indices, label_map)
		
		print("Kept class names:", keep_classes)
		print("Original indices in full dataset:", keep_class_idxs)
		print("Label remapping:", label_map)
		print("Number of classes for model:", numFolders)

		train_data = train_subset
	
	best_config = None
	best_config_score = float("inf")
	for config in hyperparameter_configs:
	
		fold_data = []
		fold_scores = []
		for fold, (train_idx, test_idx) in enumerate(kf.split(train_data)):
			train_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(train_idx))
			test_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(test_idx))

			# Use GPU if available
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

			model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

			if numFolders > -1:
				num_classes = numFolders
			else:
				num_classes = len(train_data.classes)
			model.fc = nn.Linear(model.fc.in_features, num_classes)

			for name, param in model.named_parameters():
				if not name.startswith("fc"):
					param.requires_grad = False

			torch.backends.cudnn.benchmark = True

			model = model.to(device)

			# Define optimizer and loss functions
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.Adam(model.fc.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

			best_avg_test_loss = float('inf')
			epoch_data = []

			early_stop_count = 0
			early_stop_condition = 3

			# Main epoch training loop
			for epoch in range(num_epochs):
				model.train()
				train_loss_sum = 0
				for images, labels in train_loader:
					images, labels = images.to(device), labels.to(device)

					optimizer.zero_grad()
					outputs = model(images)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()

					train_loss_sum += loss.item()
			
				# Switch from training mode to testing mode to evaluate this particular fold's performance
				model.eval()
				correct = 0
				total = 0
				test_loss_sum = 0
			
				for images, labels in test_loader:
					images, labels = images.to(device), labels.to(device)
					optimizer.zero_grad()
					outputs = model(images)
					loss = criterion(outputs, labels)

					test_loss_sum += loss.item()

					_, preds= torch.max(outputs, 1)

					correct += (preds == labels).sum().item()
					total += labels.size(0)
				
				avg_test_loss = test_loss_sum / len(test_loader)

				epoch_data.append(avg_test_loss)

				if avg_test_loss < best_avg_test_loss:
					best_avg_test_loss = avg_test_loss
					early_stop_count = 0
				else:
					early_stop_count += 1
					if early_stop_count >= early_stop_condition:
						break
			
			fold_data.append(epoch_data)
			fold_scores.append(best_avg_test_loss)

			print(f"Fold {fold+1} Eval Score: {best_avg_test_loss}")

		config_score = sum(fold_scores) / len(fold_scores)
		print(f"Avg score for config {config}: {config_score}")
		graphFoldData(fold_data)

		if config_score < best_config_score:
			best_config_score = config_score
			best_config = config

	print(f"Selected config {best_config} for evaluation. Score: {best_config_score}")
	
	return best_config

def trainModel(numFolders=-1, num_epochs=20, params={"lr": 1e-3, "weight_decay": 1e-4}):
	# Random image tranformations to increase sample variety
	train_transforms = transforms.Compose([
		transforms.RandomHorizontalFlip(p=0.5),
    	transforms.RandomRotation(degrees=20),
		transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
		transforms.RandomGrayscale(p=0.1),
		transforms.ColorJitter(
			brightness=0.2,
			contrast=0.2,
			saturation=0.2,
			hue=0.1
		),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	train_data = datasets.ImageFolder("./train_test_data/train/images", transform=train_transforms)

	numFolders = min(numFolders, len(train_data.classes))
	if numFolders > -1:
		# Selecting a subset of the total bird classes. We allow this so that we can test training smaller models that don't take nearly as long to train.
		all_classes = sorted(train_data.classes, key=lambda x: int(x))
		keep_classes = all_classes[:numFolders]

		keep_class_idxs = [train_data.class_to_idx[c] for c in keep_classes]
		subset_indices = [i for i, (_, label) in enumerate(train_data.samples)
						if label in keep_class_idxs]
		
		label_map = {old: new for new, old in enumerate(keep_class_idxs)}

		class SubsetWithRemap(torch.utils.data.Dataset):
			def __init__(self, dataset, indices, label_map):
				self.dataset = dataset
				self.indices = indices
				self.label_map = label_map

			def __len__(self):
				return len(self.indices)

			def __getitem__(self, idx):
				image, label = self.dataset[self.indices[idx]]
				return image, self.label_map[label]

		train_subset = SubsetWithRemap(train_data, subset_indices, label_map)
		
		print("Kept class names:", keep_classes)
		print("Original indices in full dataset:", keep_class_idxs)
		print("Label remapping:", label_map)
		print("Number of classes for model:", numFolders)

		train_data = train_subset
        
	train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

	# Use GPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

	if numFolders > -1:
		num_classes = numFolders
	else:
		num_classes = len(train_data.classes)
	model.fc = nn.Linear(model.fc.in_features, num_classes)

	for name, param in model.named_parameters():
		if not name.startswith("fc"):
			param.requires_grad = False

	torch.backends.cudnn.benchmark = True

	model = model.to(device)

	# Define optimizer and loss functions
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.fc.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

	best_avg_loss = float("inf")

	early_stop_count = 0
	early_stop_condition = 3
	
	# Main epoch training loop
	for epoch in range(num_epochs):
		model.train()
		loss_sum = 0
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			loss_sum += loss.item()

		avg_loss = loss_sum / len(train_loader)
		
		print(
			f"Epoch {epoch+1:02d} Avg Loss: {avg_loss:.4f}"
		)

		if avg_loss < best_avg_loss:
			best_avg_loss = avg_loss
			early_stop_count = 0
		else:
			early_stop_count += 1
			if early_stop_count >= early_stop_condition:
				break

	torch.save(model.state_dict(), f"./models/birdML_{numFolders}_birds.pth")

if __name__ == "__main__":
	# Arguments can be provided to choose whether or not to run k-fold cross-validation, choose the number of bird classes to train on, and choose the number of epochs for the train.
	if len(sys.argv) > 3:
		numClasses = int(sys.argv[2])
		numEpochs = int(sys.argv[3])

		if sys.argv[1] == "-kf":
			best_params = kFoldEval(numClasses, numEpochs)
			trainModel(numClasses, numEpochs, best_params)
	elif len(sys.argv) > 2:
		numClasses = int(sys.argv[1])
		numEpochs = int(sys.argv[2])

		trainModel(numClasses, numEpochs, {"lr": 1e-3, "weight_decay": 1e-4})
	else:
		best_params = kFoldEval()
		trainModel(params=best_params)