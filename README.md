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
git clone https://github.com/your-username/squeezenet-image-classification.git
cd squeezenet-image-classification
