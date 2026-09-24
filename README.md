# 🚗 EV Car Price Prediction using Ridge Regression

## 📌 Project Overview

This project focuses on predicting the **price of Electric Vehicles (EVs) in India** using **Machine Learning**.

The project implements **Ridge Regression** with a complete preprocessing pipeline that handles both categorical and numerical features.

Categorical features such as **Brand** and **Model** are processed using **One-Hot Encoding**, while numerical features such as **Range, Power, and Battery** are standardized using **StandardScaler**.

Different Ridge Regression regularization values (`alpha`) are tested and the model is evaluated using **MAE, RMSE, and R² Score**.

---

## 🎯 Objectives

* Analyze an Electric Vehicle dataset.
* Prepare categorical and numerical features for Machine Learning.
* Predict EV prices based on vehicle specifications.
* Implement Ridge Regression.
* Experiment with different Ridge regularization values.
* Evaluate model performance using regression metrics.
* Build a reusable Machine Learning preprocessing pipeline.

---

## 📊 Dataset

The project uses an EV dataset containing information about Electric Vehicles in India.

### Dataset Features

| Feature   | Description                       |
| --------- | --------------------------------- |
| `Brand`   | Brand/manufacturer of the EV      |
| `Model`   | Model name of the EV              |
| `Price`   | Price of the EV — Target Variable |
| `Range`   | Driving range of the vehicle      |
| `Power`   | Power of the vehicle              |
| `Battery` | Battery capacity                  |

### Target Variable

```text
Price
```

### Input Features

```text
Brand
Model
Range
Power
Battery
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 🤖 Machine Learning Algorithm

### Ridge Regression

Ridge Regression is a linear regression algorithm that uses **L2 regularization** to reduce model complexity.

The project tests multiple values of the regularization parameter:

```python
alphas = [0.01, 0.1, 1, 10, 100]
```

This allows the performance of the model to be analyzed with different levels of regularization.

---

## 🔄 Machine Learning Workflow

```text
                 EV Dataset
                     │
                     ▼
              Load Dataset
                     │
                     ▼
             Data Exploration
                     │
                     ▼
            Feature Selection
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Categorical             Numerical
      Features                Features
          │                     │
          ▼                     ▼
  One-Hot Encoding       Standard Scaling
          │                     │
          └──────────┬──────────┘
                     ▼
              Preprocessing
                  Pipeline
                     │
                     ▼
             Train-Test Split
                     │
                     ▼
             Ridge Regression
                     │
                     ▼
          Test Different Alpha
                 Values
                     │
                     ▼
               Predictions
                     │
                     ▼
             Model Evaluation
                     │
                     ▼
              MAE / RMSE / R²
```

---

## 🧹 Data Preprocessing

The dataset is divided into:

### Categorical Features

```python
categorical = ["Brand", "Model"]
```

These features are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Numerical Features

```python
numerical = ["Range", "Power", "Battery"]
```

These features are standardized using:

```python
StandardScaler()
```

### Column Transformer

Both preprocessing operations are combined using `ColumnTransformer`:

```python
preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical),

    ("numerical",
     StandardScaler(),
     numerical)
])
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing data using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The split consists of:

* **80% Training Data**
* **20% Testing Data**

---

## 🔗 Machine Learning Pipeline

The project uses a Scikit-learn `Pipeline` to combine preprocessing and Ridge Regression.

```python
model = Pipeline([
    ("preprocessing", preprocessor),
    ("ridge", Ridge(alpha=alpha))
])
```

This provides a structured workflow where preprocessing is automatically applied before training and prediction.

---

## 📈 Model Evaluation

The model uses three regression evaluation metrics.

### 1. Mean Absolute Error — MAE

MAE measures the average absolute difference between the actual and predicted prices.

```python
mean_absolute_error(y_test, test_pred)
```

---

### 2. Root Mean Squared Error — RMSE

RMSE measures the square root of the average squared prediction error.

```python
np.sqrt(mean_squared_error(y_test, test_pred))
```

---

### 3. R² Score

R² measures how well the model explains the variation in the target variable.

```python
r2_score(y_test, test_pred)
```

---

## 🧪 Ridge Regression Parameters

The project evaluates the following values of `alpha`:

| Alpha |
| ----: |
|  0.01 |
|   0.1 |
|     1 |
|    10 |
|   100 |

The purpose is to compare how different regularization strengths affect the Ridge Regression model.

---

## 📋 Results

The notebook is structured to generate a results table containing:

| Alpha | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
| ----: | --------: | ---------: | -------: | -------: | --------: | ------: |
|  0.01 |         — |          — |        — |        — |         — |       — |
|   0.1 |         — |          — |        — |        — |         — |       — |
|     1 |         — |          — |        — |        — |         — |       — |
|    10 |         — |          — |        — |        — |         — |       — |
|   100 |         — |          — |        — |        — |         — |       — |

> **Note:** The current notebook contains the evaluation structure, but the training metrics (`Train MAE`, `Train RMSE`, and `Train R²`) are referenced before they are calculated. Therefore, numerical results are not included here rather than being fabricated.

---

## 📁 Project Structure

```text
ML-Task-9/
│
├── ev_car_India_dataset.csv
│
├── EV_Car_Price_Prediction.ipynb
│
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Rabinson-20/ML-Task-9.git
```

### 2. Navigate to the Project

```bash
cd ML-Task-9
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebook

Open:

```text
EV_Car_Price_Prediction.ipynb
```

Run the cells sequentially to perform:

1. Dataset loading
2. Data exploration
3. Feature selection
4. Data preprocessing
5. Train-test splitting
6. Ridge Regression
7. Prediction
8. Model evaluation
9. Results comparison

---

## 💡 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Data loading with Pandas
* Dataset exploration
* Missing-value checking
* Feature selection
* Categorical feature encoding
* Numerical feature scaling
* Train-test splitting
* Scikit-learn `ColumnTransformer`
* Scikit-learn `Pipeline`
* Ridge Regression
* Hyperparameter experimentation
* Regression evaluation
* MAE
* RMSE
* R² Score

---

## 🚀 Future Improvements

The project can be extended by:

* Increasing the size of the EV dataset.
* Performing detailed Exploratory Data Analysis.
* Adding more vehicle features.
* Using `GridSearchCV` for automatic hyperparameter tuning.
* Applying cross-validation.
* Comparing Ridge Regression with other algorithms such as:

  * Linear Regression
  * Lasso Regression
  * Random Forest Regression
  * Gradient Boosting
* Creating data visualizations for EV price and specifications.
* Developing an interactive EV price prediction application using **Streamlit**.
* Deploying the Machine Learning model as a web application.

---

## 📌 Conclusion

This project demonstrates how Machine Learning can be applied to **predict Electric Vehicle prices** using vehicle specifications such as **brand, model, range, power, and battery capacity**.

By combining **One-Hot Encoding, Standard Scaling, ColumnTransformer, Pipeline, and Ridge Regression**, the project provides a structured approach to building a regression model for EV price prediction.

---

## 👨‍💻 Author

### **SAMRABINSON P**

BCA Student | Aspiring Full Stack Developer | Machine Learning Enthusiast
