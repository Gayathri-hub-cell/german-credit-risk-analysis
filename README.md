# 🏦 German Credit Risk Analysis

A machine learning project that predicts whether a loan applicant is a **credit risk** or not, based on financial and personal details. Built with Python and scikit-learn.

## 📊 Overview

Banks need to decide who is likely to repay a loan. This project uses the German Credit dataset to build a **binary classification** model that predicts an applicant's risk level from 20 features such as credit history, loan amount, employment duration, and age.

## 🔎 What the Notebook Does

- Loads and explores the dataset (size, summary, class balance)

- Visualizes key patterns (age distribution, loan amount vs. risk)

- Prepares the data by encoding text columns into numbers

- Trains and compares two models: **Logistic Regression** and **Random Forest**

- Evaluates them with accuracy, a confusion matrix, and a classification report

- Identifies the most important features driving the prediction

## 🛠️ Built With

- **Language:** Python

- **Libraries:** pandas, NumPy, Matplotlib, seaborn, scikit-learn

- **Environment:** Jupyter Notebook

## ▶️ How to Run

1. Open `German_Credit_Risk_Analysis.ipynb` in VS Code, Jupyter, or Google Colab.

2. Install the libraries: `pip install pandas numpy matplotlib seaborn scikit-learn`

3. Run all cells top to bottom. The dataset loads automatically from this repository.

## 📁 Dataset

`german_credit_data_biased_training.csv` — the German Credit dataset (IBM sample), included in this repo.

## 📝 Notes & Learnings

This project walks through the full machine-learning workflow end to end. The dataset is a *biased* training sample, so a real-world model would need careful fairness checks — particularly around sensitive features like `Sex` and `ForeignWorker`.
