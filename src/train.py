import sys
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import ResNet50_Weights
from torch import nn, optim
import torch
from sklearn.model_selection import KFold
from test import test

def trainModel(numFolders=-1, num_epochs=20):
	# Standard transform pipeline for ResNet input
	train_transforms = transforms.Compose([

		#Data augmentation
		transforms.RandomHorizontalFlip(p=0.5),
    	transforms.RandomRotation(degrees=20),
		transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
		transforms.RandomGrayscale(p=0.1),
		transforms.ColorJitter(                     # Random brightness/contrast/saturation
			brightness=0.2,
			contrast=0.2,
			saturation=0.2,
			hue=0.1
		),

		transforms.ToTensor(),					   # Convert PIL -> Tensor
		transforms.Normalize(						# Normalize same as ImageNet
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	kf = KFold(n_splits=5, shuffle=True, random_state=42)

	train_data = datasets.ImageFolder("./train_test_data/train/images", transform=train_transforms)

	numFolders = min(numFolders, len(train_data.classes))
	if numFolders > -1:
		# Selecting a subset of the total bird classes

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

		#train_subset = Subset(train_data, subset_indices)
		train_subset = SubsetWithRemap(train_data, subset_indices, label_map)
		
		print("Kept class names:", keep_classes)
		print("Original indices in full dataset:", keep_class_idxs)
		print("Label remapping:", label_map)
		print("Number of classes for model:", numFolders)

		train_data = train_subset

		# train_loader = DataLoader(train_subset, batch_size=32, sampler=SubsetRandomSampler(train_idx))
		# test_loader = DataLoader(train_subset, batch_size=32, sampler=SubsetRandomSampler(test_idx))
        
	best_fold=0
	best_fold_score=0
		
	for fold, (train_idx, test_idx) in enumerate(kf.split(train_data)):
		train_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(train_idx))
		test_loader = DataLoader(train_data, batch_size=32, sampler=SubsetRandomSampler(test_idx))

		# Use GPU if available
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

		# Replace final fully connected layer with correct output size
		if numFolders > -1:
			num_classes = numFolders
		else:
			num_classes = len(train_data.classes)
		model.fc = nn.Linear(model.fc.in_features, num_classes)

		# Freeze backbone
		for name, param in model.named_parameters():
			if not name.startswith("fc"):
				param.requires_grad = False

		torch.backends.cudnn.benchmark = True

		# Replace FC
		model.fc = nn.Linear(model.fc.in_features, num_classes)

		model = model.to(device)

		# Define optimizer and loss function
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

		num_epochs = 5
		for epoch in range(num_epochs):  # increase epochs for better training
			best_avg_test_loss = 0
			early_stop_count = 0
			early_stop_condition = 20 // 6
			
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

			avg_train_loss = train_loss_sum / len(train_loader)
		
			## Switches from training mode to testing mode
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
			accuracy = correct/total

			print(
				f"Fold {fold+1} | Epoch {epoch+1:02d} | "
				f"Train Loss: {avg_train_loss:.4f} | "
				f"Test Loss: {avg_test_loss:.4f} | "
				f"Accuracy: {accuracy:.4f}"
			)

			if avg_test_loss < best_avg_test_loss:
				best_avg_test_loss = avg_test_loss
				early_stop_count = 0
			else:
				early_stop_count += 1
				if early_stop_count >= early_stop_condition:
					break
		
		max_loss=1
		min_loss=0
		norm_loss = (avg_test_loss-min_loss)/(max_loss-min_loss)
		norm_acc = accuracy
		loss_weight=0.5
		acc_weight=0.5
		overfit_weight=0.1
		overfit_penalty=max(0, avg_train_loss-avg_test_loss)
		eval_score = loss_weight*(1-norm_loss) + acc_weight*norm_acc - overfit_weight*overfit_penalty
		
		if (eval_score > best_fold_score):
			best_fold_score = eval_score
			best_fold = fold + 1
		
		torch.save(model.state_dict(), f"./models/birdML_{numFolders}_birds_{fold+1}.pth")
		
	# Evaluate best fold
	print(f"Selected fold {best_fold} for evaluation. Score: {best_fold_score}")
	test(numFolders, best_fold)

if __name__ == "__main__":
	if len(sys.argv) > 2:
		trainModel(int(sys.argv[1]), int(sys.argv[2]))
	else:
		trainModel()