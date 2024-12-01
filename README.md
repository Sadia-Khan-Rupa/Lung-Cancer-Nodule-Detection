# Image Classification Using SqueezeNet and MobileNet for Binary Classification

This repository implements two image classification models for a binary classification task using a custom dataset of images. The models are:

1. **SqueezeNet-based Model** - A lightweight CNN architecture for fast and efficient image classification.
2. **MobileNet-based Model** - A highly efficient deep neural network architecture for mobile and embedded devices, optimized for performance.

Both models are trained and evaluated on a dataset of 50x50 grayscale images, and they are capable of classifying images into two classes.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Dependencies](#dependencies)
4. [Data](#data)
5. [Model Architectures](#model-architectures)
   - [SqueezeNet Model](#squeezenet-model)
   - [MobileNet Model](#mobilenet-model)
6. [Training and Evaluation](#training-and-evaluation)
7. [Results](#results)
   

## Project Overview

This project contains two models for binary image classification:

- **SqueezeNet-based Model**: A deep convolutional network designed for efficiency and speed with a lightweight architecture.
- **MobileNet-based Model**: A lightweight deep convolutional network optimized for mobile and embedded applications. It uses depthwise separable convolutions for a more efficient computation compared to standard convolutions.

Both models are evaluated on the same dataset and aim to predict the binary class label for each input image. The project includes training, evaluation, and performance metrics like accuracy, AUC, and ROC curves.

## Getting Started


To get started with this project, clone the repository:

```bash
git clone https://github.com/Sadia-Khan-Rupa/Lung-Cancer-Nodule-Detection.git
cd Lung-Cancer-Nodule-Detection
```
---

## **Prerequisites**
This project requires Python and the following libraries:

TensorFlow (Keras)
NumPy
Pandas
Scikit-learn
Matplotlib
h5py
You can install the dependencies by running:

pip install -r requirements.txt


### Dependencies

Python >= 3.7
TensorFlow >= 2.0
NumPy
Pandas
Scikit-learn
Matplotlib
h5py
For the full list of dependencies, refer to requirements.txt.

---
## Data

This project expects the data in an HDF5 format (dataset.h5). The dataset should contain two groups:

X: The images (50x50 pixels, grayscale)

Y: The labels (binary, one-hot encoded)

Example of loading data:
train_images, tr_labels = load_hdf_dataset('train_again')

val_images, val_labels = load_hdf_dataset('val_again')

You can replace 'train_again' and 'val_again' with the path to your dataset.


---
## **Model Architectures**

**SqueezeNet Model**
The SqueezeNet model is a lightweight deep convolutional network used for image classification. It consists of several convolutional layers and fire modules, which use a combination of 1x1 and 3x3 convolutions to reduce the computational cost. The architecture includes:

**Convolutional layers with ReLU activation**
Fire modules (compression with 1x1 convolutions and expansion with 1x1 and 3x3 convolutions)
Global Average Pooling
Dropout and softmax output layer
MobileNet Model
The MobileNet model is designed for mobile and embedded applications, emphasizing speed and efficiency. It uses depthwise separable convolutions, which break down the standard convolution into two layers—one for filtering and one for combining outputs. This results in a much smaller model that retains good performance.

**The architecture of MobileNet consists of:**

Depthwise separable convolutions
Global Average Pooling
Fully connected layers for classification
Softmax output layer

You can build and train MobileNet with the following code:

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def build_mobilenet(input_shape):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

---
## **Training and Evaluation**

The models are trained using the following parameters:

Optimizer: Adam (learning rate: 0.0001)
Loss function: Categorical Crossentropy
Metrics: Accuracy
Batch size: 32
Epochs: 10
Class weights are applied to address class imbalance.

Example Training for SqueezeNet:
model_dw.fit(x=train_images,
             y=tr_labels,
             batch_size=32,
             epochs=10,
             validation_data=(val_images, val_labels),
             class_weight=wt,
             shuffle=True)
Similarly, you can train your MobileNet model by following the same training loop.

---

## **Performance Metrics:**
Accuracy and loss for training and validation sets
ROC curve and AUC score for model evaluation
Results

After training, the models achieve the following performance:

**SqueezeNet Accuracy: ~84.1% (Training), ~83.3% (Validation)**
**MobileNet Accuracy: ~70.1% (Training), ~64.0% (Validation)**
**AUC Score: 0.83 (for SqueezeNet) and (Add results for MobileNet here)**

### ROC Curve:

The models' ROC curves show the trade-off between true positive rate and false positive rate.

plt.plot(fpr[1], tpr[1])

---
## 📞 **Contact**

Email: khanrupasadia@gmail.com
LinkedIn: [Sadia LinkedIn](https://www.linkedin.com/in/sadia-khan-rupa/)

