# 🧠 Handwritten Digit Prediction  
**with TensorFlow & Pygame**

A simple graphical application built using **Pygame**, allowing users to draw a digit (0–9) and have it recognized by a **Convolutional Neural Network (CNN)** trained on the **MNIST** dataset using **TensorFlow/Keras**.

---

## ✨ Features

- 🖊️ Interactive drawing interface using Pygame  
- 🧠 CNN model trained to recognize handwritten digits  
- 📈 Optional model training on MNIST  
- 💾 Save/load model functionality  
- 🔮 Real-time on-screen prediction

---

## 📦 Requirements

Python 3.x and the following libraries:

- `pygame`  
- `tensorflow`  
- `numpy`  
- `matplotlib` *(only for training)*

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

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

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone git@github.com:JanBanasik/Tensorflow_number_prediction.git
cd Tensorflow_number_prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (optional)

If `cnn_mnist.keras` is missing in `Tensorflow_number_prediction/`, train the model:

```bash
python -m Tensorflow_implementation.Utils.trainModel
```

### 4. Run the Pygame application

```bash
python -m Tensorflow_implementation.main.main
```

---

## 🧑‍💻 Usage Instructions

Once the app launches:

- 🖊️ Draw: Hold **left mouse button** and drag  
- 🧽 Erase: Hold **right mouse button**  
- 🔮 Predict: Press the **P** key  
- ♻️ Clear Board: Press the **C** key  
- ✅ Prediction appears in the **top-left corner**

---

## 📄 License

This project is licensed under the **MIT** License.  
See the `LICENSE` file for more details.
