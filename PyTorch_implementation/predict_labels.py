import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def preprocess_grid(grid):
    grid_tensor = torch.tensor(np.array(grid), dtype=torch.float32).unsqueeze(0)
    return grid_tensor


def predict(grid):
    image_tensor = preprocess_grid(grid)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    # Predykcja
    with torch.no_grad():
        predictions = model(image_tensor)
        predicted_label = torch.argmax(predictions, dim=1).item()

    return predicted_label


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    #print(type(train_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    examples = next(iter(train_loader))

    # Rozdziel obrazki (examples[0]) i ich etykiety (examples[1])
    images, labels = examples

    # Wyświetl pierwszy obrazek w batchu
    plt.imshow(images[0].squeeze(), cmap='Greys')
    plt.axis('off')
    plt.show()
    # for i in images[0]:
    #     print(*i)
    # # Wypisz odpowiadającą mu labelkę
    # print(f"Label for the first image: {labels[0].item()}")
