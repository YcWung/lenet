import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import get_train_test_datasets
from lenet import LeNet
import matplotlib.pyplot as plt

# Define hyperparameters
batch_size = 64

# Download and load dataset
_, test_dataset = get_train_test_datasets()
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print image size
print(f"Image size: {test_dataset[0][0].shape}")

# plot the first 5 images in the test set
fig = plt.figure(figsize=(5, 1))
fig.suptitle('Test Images', fontsize=20)
for i in range(5):
    image, label = test_dataset[i]
    fig.add_subplot(1, 5, i+1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')

plt.savefig('test_images.png')

# Define model and load weights
model = LeNet()
model.load_state_dict(torch.load('lenet.pth'))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy of the model on the {len(test_dataset)} test images: {100 * correct / total}%")
