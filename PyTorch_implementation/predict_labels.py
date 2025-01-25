import numpy as np
import torch.nn as nn
import torch


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
    model.load_state_dict(torch.load('mnist_model_adam_optimizer_binaryCodedToMinusOne.pth'))
    model.eval()

    with torch.no_grad():
        predictions = model(image_tensor)
        predicted_label = torch.argmax(predictions, dim=1).item()

    return predicted_label
