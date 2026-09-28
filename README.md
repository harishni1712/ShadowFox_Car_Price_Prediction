# Car Selling Price Prediction and Analysis

## Project Overview

This project was developed as part of the **ShadowFox Machine Learning Engineer Internship**.

The objective of this project is to develop a Machine Learning model that predicts the approximate selling price of a used car based on different car-related features.

The project includes data preprocessing, exploratory data analysis, feature engineering, model training, model comparison, feature importance analysis, and deployment using Streamlit.

---

## Problem Statement

Develop an ML model for car selling price prediction and analysis.

The system predicts an approximate selling price based on features such as:

- Fuel Type
- Years of Service
- Showroom Price
- Number of Previous Owners
- Kilometers Driven
- Seller Type
- Transmission Type

---

## Dataset

The dataset contains information about used cars and their selling prices.

### Features

- `Car_Name` – Name of the car
- `Year` – Manufacturing year
- `Present_Price` – Current showroom price
- `Kms_Driven` – Kilometers driven
- `Fuel_Type` – Petrol, Diesel, or CNG
- `Seller_Type` – Dealer or Individual
- `Transmission` – Manual or Automatic
- `Owner` – Number of previous owners

### Target

- `Selling_Price` – Selling price of the car in lakhs

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Feature Importance Analysis
   ↓
Model Saving
   ↓
Streamlit Deployment