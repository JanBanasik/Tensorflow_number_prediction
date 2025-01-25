import numpy as np
import torch.nn as nn
import torch


class LabelPredictor:
    path = "mnist_model_adam_optimizer_binaryCodedToMinusOne.pth"

    def __init__(self):
        self.model = self.buildModel()

    def buildModel(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        model.load_state_dict(torch.load(self.path))
        return model

    def preprocess_grid(self, grid):
        grid_tensor = torch.tensor(np.array(grid), dtype=torch.float32).unsqueeze(0)
        return grid_tensor

    def predict(self, grid):
        image_tensor = self.preprocess_grid(grid)

        self.model.eval()

        with torch.no_grad():
            predictions = self.model(image_tensor)
            predicted_label = torch.argmax(predictions, dim=1).item()
        return predicted_label
