To load the trained model:

model = models.resnet50(weights=None)  
model.fc = nn.Linear(model.fc.in_features, 4)  
model.load_state_dict(torch.load('resnet50_trained_model.pth'))  
model = model.to(device)  