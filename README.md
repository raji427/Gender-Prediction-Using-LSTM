# LSTM Baby Names Gender Predictor

Predict whether a baby name is **male** or **female** using a character-level LSTM deep learning model trained on the US NationalNames dataset.

## Overview

This project demonstrates building a sequence classification model using LSTM (Long Short-Term Memory) neural networks to predict the gender label for baby names. Each name is encoded as a sequence of characters, which are mapped to integers and fed into an LSTM for binary classification. The model outputs a probability, which is used to classify the name as male or female.

## Features

- **Character-level encoding:** Names are split into lowercase characters and mapped to unique integers.
- **Sequence padding:** All sequences are padded to a fixed length (10) for model input.
- **Label encoding:** Gender labels are encoded as 0 (Female) and 1 (Male).
- **LSTM model architecture:** Includes input, embedding, stacked LSTM layers, and a dense output unit with sigmoid activation.
- **Training and validation accuracy visualization:** Plots accuracy curves as the model trains.
- **Custom gender prediction function:** Pass any name string for inference.
- **Clear classification rule:** If the predicted value is less than 0.5, the result is **Female**; otherwise, it is **Male**.

## Getting Started

### Prerequisites

Python 3.x and the following libraries:
- pandas
- numpy
- matplotlib
- keras
- tensorflow

Install them using pip:
pip install pandas numpy matplotlib keras tensorflow


### Dataset

Download the [NationalNames.csv dataset](https://www.kaggle.com/datasets/kaggle/us-baby-names) and place it in your project directory.

### Usage

1. **Run the training and evaluation script** (e.g., `lstm_baby_names.py`).

2. The script will:
   - Load and preprocess the names dataset
   - Encode names and genders
   - Visualize name length distribution
   - Train an LSTM model for gender classification
   - Display training and validation accuracy curves

3. **Predict gender for a new name:**

Use the following function in your script to get predictions:

def predict_gender(name):
  test_name = name.lower()
  seq = [vocab[i] for i in test_name]
  x_test = pad_sequences([seq], 10)
  y_pred = my_model.predict(x_test)
  if y_pred < 0.5:
  return "Female"
  else:
  return "Male"

Example:
print(predict_gender('Devi')) # Output: Female
print(predict_gender('John')) # Output: Male

### How It Works

- Names are grouped and averaged by gender, then converted into integer sequences.
- Sequences are padded to a maximum length of 10 characters.
- The model uses an embedding layer and two stacked LSTM layers with a sigmoid output for binary classification.
- Classification is based on the sigmoid function output:
  - **Output < 0.5:** Female
  - **Output >= 0.5:** Male

### Visualization

- Plots histograms of name length distribution.
- Shows training and validation accuracy over epochs.

## Customization

- Modify the sequence length or model architecture to improve performance.
- Extend the character set or dataset for language-specific names.

## Limitations

- Model accuracy depends on the US National Names dataset; may be less accurate with rare or international names.

Install them using pip:

