# ✍️ Handwritten Character Recognition using CNN

This project demonstrates how to recognize **handwritten English characters (A–Z)** using a **Convolutional Neural Network (CNN)** implemented with TensorFlow/Keras. It uses a labeled dataset of grayscale character images to train an image classifier capable of accurately predicting letters written by hand.

---

## 📌 Features

- Recognizes 26 uppercase letters (A–Z)
- Achieves high accuracy with a simple CNN
- Trains on 28x28 grayscale images
- Saves trained model as `.h5` for future predictions
- Visualizes results with confusion matrix and metrics

---

## 📁 Dataset

- 📄 **Name**: A_Z Handwritten Data
- 🧾 **Format**: `.npy` (NumPy array)
- 🔤 **Classes**: 26 (A to Z)
- 📐 **Image size**: 28x28 pixels (grayscale)

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) and convert it into `.npy` format or use a preprocessed version.

---

## 🧠 Model Architecture

```
Input: 28x28x1 grayscale image

[Conv2D]      32 filters, 3x3 kernel, ReLU
[MaxPooling2D] 2x2
[Conv2D]      64 filters, 3x3 kernel, ReLU
[MaxPooling2D] 2x2
[Flatten]
[Dense]       128 neurons, ReLU
[Dense]       26 neurons, Softmax (for A-Z)
```

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

### 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/p-v-srinag/Hand-written-recognition-using-CNN.git
   cd Hand-written-recognition-using-CNN
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow matplotlib seaborn scikit-learn numpy
   ```

3. **Add the dataset**:
   - Place the file `A_Z Handwritten Data.npy` in the project folder.

4. **Run the notebook**:
   Open this file in Jupyter:
   ```
   HandWritten-Character-Recognition-ML-Model-Using-CNN[1].ipynb
   ```

---

## 🧪 Model Evaluation

- **Accuracy**: ~98% on the test set
- **Evaluation Tools**:
  - Classification Report
  - Confusion Matrix (using `seaborn.heatmap`)
- **Model File**:
  ```
  character_recognition_model.h5
  ```

---

## 💾 Inference Example

```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("character_recognition_model.h5")

# Sample image: shape must be (1, 28, 28, 1)
prediction = model.predict(sample_image)
predicted_class = chr(np.argmax(prediction) + ord('A'))
print("Predicted character:", predicted_class)
```


---

## 🙌 Acknowledgements

- Dataset: [A-Z Handwritten Alphabet (Kaggle)](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
- TensorFlow/Keras documentation
- Scikit-learn utilities for metrics and evaluation

---

## 🌟 Show your support

If you like this project, please ⭐️ the repo to help others discover it!

---
