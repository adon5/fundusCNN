To load the trained model:

model = models.resnet50(weights=None)  
model.fc = nn.Linear(model.fc.in_features, 4)  
model.load_state_dict(torch.load('resnet50_trained_model.pth'))  
model = model.to(device)  

Performance Metrics:

Accuracy (% correct):
(TP + TN) / (TP + TN + FP + FN)
Recall (Sensitivity):
TP / (TP + FN)
Precision (PPV):
TP / (TP + FP)
Specificity (NPV):
TN / (TN + FP)
F1-Score:
2 * (Precision * Recall) / (Precision + Recall)