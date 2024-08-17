import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('resnet50_fundus_trained.pth'))
model.to(device)
model.eval()

output_classes = {0: "cataract", 1: "diabetic_retinopathy", 2: "glaucoma", 3: "normal"}

# Using the same transforms as training
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = "./data"


class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = list(output_classes.values())
        self.class_to_idx = {cls_name: idx for idx, cls_name in output_classes.items()}
        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir)[:200]:  # Limit to 200 images per class
                self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


dataset = FundusDataset(data_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
precision = Precision(task="multiclass", average='macro', num_classes=4).to(device)
recall = Recall(task="multiclass", average='macro', num_classes=4).to(device)
f1_score = F1Score(task="multiclass", average='macro', num_classes=4).to(device)

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        accuracy(preds, labels)
        precision(preds, labels)
        recall(preds, labels)
        f1_score(preds, labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Accuracy: {accuracy.compute().item():.4f}")
print(f"Precision: {precision.compute().item():.4f}")
print(f"Recall: {recall.compute().item():.4f}")
print(f"F1 Score: {f1_score.compute().item():.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=output_classes.values(),
            yticklabels=output_classes.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion matrix saved as 'confusion_matrix.png'")
