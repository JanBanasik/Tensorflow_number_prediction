import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Tensorflow_implementation.Models.CNN_MNIST_class import CNN_mnist

if __name__ =="__main__":

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = ((X_train / 255) > 0.5).astype(np.float32)
    X_test = ((X_test / 255 ) > 0.5).astype(np.float32)

    X_train = X_train[..., tf.newaxis]
    X_test = X_test[..., tf.newaxis]
    print(X_train.shape)

    random_index = np.random.randint(0, len(X_train))
    plt.imshow(X_train[random_index])

    model = CNN_mnist()

    model.compile(optimizer="adam",
                 loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=["accuracy"])

    model.fit(X_train,
             y_train,
             epochs=5,
             batch_size=64,
             validation_split=0.1)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.2%}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    random_indices = np.random.randint(0, len(y_test), size=4)

    for index, ax in zip(random_indices, axes.ravel()):
        example_x, example_y = X_test[index], y_test[index]
        example_x_expanded = np.expand_dims(example_x, axis=0)

        logits = model.predict(example_x_expanded)
        print(logits)
        probs = tf.nn.softmax(logits)
        predicted_class = tf.argmax(probs, axis=1).numpy()[0]

        print(f"True label: {example_y}")
        print(f"Predicted class: {predicted_class}")
        print(f"Probabilities: {probs.numpy()}")

        ax.imshow(example_x)
        ax.set_title(f"Predicted class: {predicted_class} \n True label: {example_y}")

    plt.show()
    model.save("cnn_mnist.keras")
