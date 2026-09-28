# 🎓 Adult Education Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Status-In%20Development-2ea44f?style=for-the-badge" alt="Status">
</p>

<p align="center">
  <strong>📊 Machine Learning project for predicting education level using the Adult Dataset.</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-objectives">Objectives</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-pipeline">Pipeline</a> •
  <a href="#-models">Models</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 🚀 Overview

This project applies a complete **Data Science and Machine Learning workflow** to the **Adult Dataset**.

The main objective is to build a model capable of predicting an individual's **education level** using the available characteristics in the dataset.

The problem is formulated as a:

> 🎯 **Multiclass Classification problem**

Two machine-learning approaches are considered:

* 🌳 **Decision Tree**
* 🌲 **Random Forest**

---

## 🎯 Objectives

### 💼 Business Objective

> **Predict the education level of an individual.**

The target variable is:

```text
education
```

### 🤖 Data Science Objective

| Component    | Description                       |
| ------------ | --------------------------------- |
| Problem type | Multiclass Classification         |
| Target       | `education`                       |
| Features     | Other available dataset variables |
| Model 1      | Decision Tree                     |
| Model 2      | Random Forest                     |

---

## 📦 Dataset

The project uses the **Adult Dataset**.

The original `adult.data` file does not contain column names, therefore the column names are provided manually during the data-loading stage.

The dataset contains:

<div align="center">

### 📈 32,561

**Rows**

### 📊 15

**Columns**

</div>

---

## 🧠 Machine Learning Workflow

The project follows a structured Data Science pipeline:

```text
                    ┌─────────────────────┐
                    │    Adult Dataset    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Loading      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Exploration    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Cleaning       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Missing Values      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Processing  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Encoding            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Normalization       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Correlation Matrix  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Selection   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Final Dataset     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             🌳 Decision Tree      🌲 Random Forest
                    │                     │
                    └──────────┬──────────┘
                               ▼
                     🎓 Education Prediction
```

The complete preparation process documented in the project includes data loading, exploration, cleaning, missing-value management, variable removal, visualization, encoding, normalization, correlation analysis and feature selection.

---

# 🔬 Data Preparation

## 1️⃣ Data Loading

The `adult.data` file is loaded into Python.

Because the original dataset does not contain a header, the column names are provided manually.

The `skipinitialspace` parameter is used to handle spaces before values.

---

## 2️⃣ Data Exploration

The first exploration stage is used to understand the structure and dimensions of the dataset.

### Dataset dimensions

```text
Rows    → 32,561
Columns → 15
```

---

## 3️⃣ Data Cleaning

The dataset is cleaned before continuing with the Machine Learning workflow.

The project focuses particularly on the variables:

```text
education
income
```

The cleaning process is used to prepare the data for subsequent analysis.

---

## 4️⃣ Missing Values

Missing values are identified and handled during the preprocessing stage.

This helps ensure that the dataset is suitable for the following transformation and Machine Learning stages.

---

## 5️⃣ Feature Removal

Variables that are considered unnecessary or unusable for the analysis are removed.

The objective is to retain relevant information for predicting:

```text
education
```

---

## 📊 Data Visualization

Visualization is used to explore the dataset and understand the distribution and relationships between variables.

This stage helps provide a visual understanding of the data before applying transformations and Machine Learning algorithms.

---

# 🔄 Feature Engineering

## 🔢 Encoding

Categorical variables need to be transformed into numerical representations before being used by Machine Learning algorithms.

The project therefore includes a dedicated **transformation and encoding** stage.

---

## 📏 Normalization

Normalization is applied during the data-preparation process to transform numerical features to a comparable scale where appropriate.

---

## 🔥 Correlation Matrix

A correlation matrix is generated to analyze relationships between variables.

It is also used as part of the process of identifying relevant features.

---

## 🎯 Feature Selection

Feature selection is performed to identify the characteristics that are relevant to the prediction problem.

The final objective is to obtain a prepared dataset containing useful information for predicting:

```text
education
```

---

# 🤖 Models

The project uses two Machine Learning approaches for the multiclass classification task.

## 🌳 Decision Tree

A **Decision Tree** is used to perform multiclass classification.

```text
Input Features
      │
      ▼
 ┌─────────────┐
 │ Decision    │
 │    Tree     │
 └──────┬──────┘
        │
        ▼
 Education Class
```

---

## 🌲 Random Forest

A **Random Forest** is also used as a classification algorithm.

```text
              ┌── Decision Tree 1 ──┐
              │                      │
Input ────────┼── Decision Tree 2 ──┼──► Prediction
              │                      │
              └── Decision Tree N ──┘
```

Both algorithms are identified in the project as the models for the multiclass classification task.

---

# 🗂️ Project Structure

A possible organization for the project is:

```text
Adult-Education-Prediction/
│
├── 📁 data/
│   ├── adult.data
│   └── adult.names
│
├── 📁 notebooks/
│   └── education_analysis.ipynb
│
├── 📁 src/
│   ├── data_preparation.py
│   ├── visualization.py
│   ├── feature_selection.py
│   └── models.py
│
├── 📁 images/
│   ├── visualization.png
│   └── correlation_matrix.png
│
├── 📄 README.md
└── 📄 requirements.txt
```

---

# 🛠️ Technologies

<p align="center">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="60">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="60">

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="60">

</p>

| Technology      | Purpose              |
| --------------- | -------------------- |
| 🐍 Python       | Programming language |
| 🐼 Pandas       | Data manipulation    |
| 🔢 NumPy        | Numerical operations |
| 📊 Matplotlib   | Data visualization   |
| 🤖 Scikit-learn | Machine Learning     |

---

# 📋 Project Steps

|  # | Step                      | Status |
| -: | ------------------------- | :----: |
|  1 | Import libraries          |    ✅   |
|  2 | Load dataset              |    ✅   |
|  3 | Explore dataset           |    ✅   |
|  4 | Clean data                |    ✅   |
|  5 | Handle missing values     |    ✅   |
|  6 | Remove unusable variables |    ✅   |
|  7 | Visualization             |    ✅   |
|  8 | Encoding                  |    ✅   |
|  9 | Normalization             |    ✅   |
| 10 | Correlation matrix        |    ✅   |
| 11 | Feature selection         |    ✅   |
| 12 | Create final dataset      |    ✅   |
| 13 | Decision Tree             |   🔄   |
| 14 | Random Forest             |   🔄   |

The first twelve stages correspond to the workflow documented in the project PDF.

---

# 📈 Results

> ⚠️ **Note:** The current project documentation does not provide final model-performance metrics such as accuracy, precision, recall, F1-score or AUC. These should be added after the models are trained and evaluated.

Recommended evaluation section:

```text
Model              Accuracy    Precision    Recall    F1-Score
----------------------------------------------------------------
Decision Tree         --          --          --         --
Random Forest         --          --          --         --
```

---

# 🚀 Future Improvements

Possible future additions to the project include:

* 📊 Add model evaluation metrics
* 🔥 Add confusion matrices
* 📈 Compare Decision Tree and Random Forest results
* 🎯 Analyze feature importance
* ⚙️ Perform hyperparameter tuning
* 📉 Add ROC/AUC analysis where appropriate
* 💾 Save the trained model
* 🌐 Build a small prediction interface

---

# 👨‍💻 Author

### Wassim Saadli

🎓 Computer Science Engineering Student
🐍 Data Science & Machine Learning
☁️ Cloud & DevSecOps

---

<p align="center">

### ⭐ If you find this project useful, consider giving it a star!

**Built with Python • Data Science • Machine Learning**

</p>
