import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def show_images(images, labels, num_images=6):
    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        image = images[i].squeeze().numpy()  # Usunięcie kanału i konwersja na numpy
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()


def calculate_accuracy(loader, trainedModel):
    correct = 0
    total = 0
    trainedModel.eval()
    with torch.no_grad():
        for images, labels in loader:
            output = trainedModel(images)
            _, predicted = torch.max(output.data, 1)
            correct += np.array(predicted == labels).sum().item()
            total += labels.size(0)
    trainedModel.train()
    return correct / total


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        feature, target = data
        pred = model(feature)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    training_accuracy = calculate_accuracy(train_loader, model)
    testing_accuracy = calculate_accuracy(test_loader, model)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, "
          f"Train Accuracy: {training_accuracy * 100:.2f}%, Test Accuracy: {testing_accuracy * 100:.2f}%")

torch.save(model.state_dict(), "mnist_model_adam_optimizer.pth")
print("Model zapisany jako 'mnist_model.pth'")
