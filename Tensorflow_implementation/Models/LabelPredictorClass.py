import numpy as np
import tensorflow as tf
from Tensorflow_implementation.Models.CNN_MNIST_class import CNN_mnist
import os

class LabelPredictor:
    path = os.path.join(os.path.dirname(__file__), "../../cnn_mnist.keras")

    def __init__(self):
        self.model = self.__buildModel()

    def __buildModel(self):

        loaded_model = tf.keras.models.load_model(self.path, custom_objects={"CNN_mnist": CNN_mnist})
        return loaded_model



    def predict(self, grid) -> tf.Tensor:
        grid = (grid > 0).astype(np.float32)
        grid = grid[..., np.newaxis]
        grid = np.expand_dims(grid, axis=0)


        logits = self.model.predict(grid)
        probs = tf.nn.softmax(logits)
        predicted_class = tf.argmax(probs, axis=1)
        return predicted_class.numpy()[0]


# tests
if __name__ == "__main__":
    # predictor = LabelPredictor()
    # print(predictor.predict(tab))
    pass