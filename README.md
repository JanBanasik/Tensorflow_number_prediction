# Handwritten Digit Prediction using TensorFlow and Pygame

### This project is a simple graphical application built with the Pygame library, allowing the user to draw a digit from 0 to 9.
### The drawn image is then processed and passed to a Convolutional Neural Network (CNN) model (built using TensorFlow/Keras), which predicts the digit. 
### A separate script allows training the model on the MNIST dataset.

---

# ✨ Features

- ### Interactive drawing interface using Pygame.
- ### CNN model (TensorFlow/Keras) trained to recognize handwritten digits (0–9).
- ### Ability to train the model from scratch on MNIST.
- ### Save/load model support.
- ### On-screen prediction feedback.

---

# 🛠 Requirements

### Python 3.x with the following libraries:

- `pygame`
- `tensorflow`
- `numpy`
- `matplotlib` (only needed during training)

### Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure
```
Tensorflow_number_prediction/
├── Tensorflow_implementation/
│   ├── main/
│   │   ├── __init__.py
│   │   └── main.py                  # Main Pygame application
│   ├── Models/
│   │   ├── __init__.py
│   │   ├── CNN_MNIST_class.py       # CNN model definition
│   │   ├── Colors.py                # Pygame color constants
│   │   ├── LabelPredictorClass.py   # Model loading and prediction class
│   │   └── Node.py                  # Grid cell representation
│   └── Utils/
│       ├── __init__.py
│       ├── cnn_mnist.keras          # Saved trained model
│       ├── draw_grid.py             # Grid drawing and UI logic
│       └── trainModel.py            # Training script
├── LICENSE
├── README.md
└── requirements.txt

```

# 🚀 How to Run

## 1. Clone the repository
```bash
git clone git@github.com:JanBanasik/Tensorflow_number_prediction.git
cd Tensorflow_number_prediction
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Train the model (optional)
### If cnn_mnist.keras is missing in Tensorflow_number_prediction/, train the model:

```bash
python -m Tensorflow_implementation.Utils.trainModel
```

## 4. Run the Pygame application
```bash
python -m Tensorflow_implementation.main.main
```

## 🧑‍💻 Usage Instructions
## Once the app launches:

- Draw: Hold left mouse button and drag.

- Erase: Hold right mouse button.

- Predict: Press the P key.

- Clear Board: Press the C key.

- Prediction result appears in the top-left corner.

# 📄 License
### This project is licensed under the [LICENSE NAME, e.g., MIT] License.
### See the LICENSE file for more details.