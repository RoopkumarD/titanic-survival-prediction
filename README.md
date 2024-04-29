# Titanic Survival Prediction Model

## Overview

This repository contains a Jupyter Notebook demonstrating the creation and training of a machine learning model to predict
survival on the Titanic dataset using a Random Forest Classifier. The notebook outlines the process of data preprocessing,
feature selection, model training, and evaluation.

## Files

- `titanic_model.ipynb`: Jupyter Notebook containing the Python code for data preprocessing, model training, and evaluation.
- `README.md`: This file, providing an overview of the project and instructions for use.
- `.csv` are train and test dataset.

## Getting Started

To run the notebook locally, follow these steps:

1. Clone this repository to your local machine using:

```
git clone https://github.com/RoopkumarD/titanic-survival-prediction.git
```

2. Navigate to the repository directory:

```
cd titanic-survival-prediction
```

3. Open the `titanic_survival_prediction.ipynb` file in Jupyter Notebook or Jupyter Lab.

4. Follow the instructions and code within the notebook to execute the data preprocessing steps, train the model, and
   evaluate its performance.

## Dependencies

The notebook requires the following Python libraries:
- pandas
- numpy
- scikit-learn
- category-encoders

You can install these dependencies using pip:

```
pip install category-encoders pandas numpy scikit-learn
```

## Dataset

The dataset used in this project is the famous Titanic dataset, which contains information about passengers aboard the RMS
Titanic, including whether they survived or not. The dataset is included in the repository as `train.csv` and `test.csv`.

## Model Training

The model is trained using a Random Forest Classifier. Key steps in model training include:
- Data preprocessing: Handling missing values, feature engineering, and encoding categorical variables.
- Feature selection: Choosing relevant features for training the model.
- Model training: Using the Random Forest Classifier to train the model.

## Evaluation

The model's performance is evaluated using accuracy score and other relevant metrics. Additionally, cross-validation
techniques are employed to ensure the robustness of the model.

## Contribution

Contributions to this repository are welcome. Feel free to open an issue or submit a pull request with any improvements or
suggestions.
