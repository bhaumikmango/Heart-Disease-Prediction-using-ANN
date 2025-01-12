# Heart Disease Prediction using Artificial Neural Network (ANN)

This repository contains a machine learning model built to predict heart disease based on various features such as age, cholesterol levels, blood pressure, and more. The model is developed using a simple Artificial Neural Network (ANN) architecture.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Project Overview

This project is built to predict the likelihood of a person having heart disease based on historical data. The data consists of several clinical features such as age, sex, blood pressure, cholesterol levels, and other relevant metrics. The goal is to build a predictive model using a deep learning algorithm (Artificial Neural Network) and evaluate its performance.

### Steps:
1. **Exploratory Data Analysis (EDA):** Visualized the relationships between various features using pair plots and correlation heatmaps.
2. **Data Preprocessing:** Scaled the features and split the data into training and testing sets.
3. **Model Training:** Used an Artificial Neural Network (ANN) with the Keras framework to train the model.
4. **Model Saving:** Saved the trained model and scaler for future predictions.

## Data Preprocessing

The dataset used is the well-known **Heart Disease UCI dataset**. It contains several attributes related to heart disease prediction. 

### Features:
- **age**: Age of the patient
- **sex**: Gender of the patient (0 = female, 1 = male)
- **cp**: Chest pain type (categorical)
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol
- **fbs**: Fasting blood sugar (binary)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (binary)
- **oldpeak**: Depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy
- **thal**: Thalassemia (categorical)
- **target**: Whether the patient has heart disease (1 = yes, 0 = no)

### Data Processing Steps:
- Separated features and target variables.
- Standardized the feature data using `StandardScaler`.
- Split the data into training and testing sets (70% training, 30% testing).

## Model Training

The Artificial Neural Network (ANN) model is implemented using Keras. The model consists of:
- **Input layer**: 13 input features
- **Hidden layers**: Two dense layers with ReLU activation
- **Output layer**: A single neuron with a sigmoid activation function for binary classification

The model was compiled using the Adam optimizer and binary cross-entropy as the loss function.

### Training:
- Trained on the training dataset for 100 epochs with a batch size of 8.
- The model was saved after training to `heart_disease_model1.h5` for future use.

## Model Evaluation

After training, the model can be evaluated on the test set using the `evaluate()` function. The test accuracy helps in understanding the model's performance on unseen data.

### Explanation:
- **Project Overview**: Describes the purpose and goal of the project.
- **Data Preprocessing**: Details the steps you followed to prepare the data.
- **Model Training**: Outlines how the ANN model was created and trained.
- **Model Evaluation**: Explains how to evaluate the model after training.
- **Files Structure**: Lists the files present in the repository.
- **How to Run**: Provides step-by-step instructions on how to clone the repo, install dependencies, train the model, and make predictions.
- **Dependencies**: Lists all the necessary Python libraries to run the project.
- **License**: Specifies that the project is licensed under the MIT License.

## To clone the repository run on CMD :
git clone https://github.com/bhaumikmango/Heart-Disease-Prediction-using-ANN
cd heart-disease-prediction
