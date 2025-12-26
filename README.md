# Dimensionality Reduction and Neural Network Modeling for Flight Price Categorization

## 📘 Overview
This repository contains the **Final Project** for  
**Predictive Analytics**.

The project develops a complete classification framework for airline ticket prices by integrating **Principal Component Analysis (PCA)** with **neural network modeling**. A fully numerical feature matrix is constructed from raw flight data, reduced using PCA, reconstructed, and then used to train neural network classifiers to categorize flights as **Cheap** or **Expensive**.

The study emphasizes dimensionality reduction, model stability, architectural tuning, and interpretability through sensitivity analysis.

---

## 🎯 Objectives
- Engineer a clean numerical feature set from raw flight data
- Apply PCA to reduce dimensionality while retaining at least 80% variance
- Reconstruct features from selected principal components
- Build and evaluate neural network classifiers for price categorization
- Compare baseline and fine-tuned neural network architectures
- Analyze model stability across multiple random train–test splits
- Interpret model behavior using sensitivity analysis

---

## 📊 Dataset Description
- **Source:** Data_Train.csv  
- **Total Records:** 10,683  
- **Final Clean Observations:** 10,682  

### Engineered Numerical Features
- Duration Minutes
- Departure Hour
- Arrival Hour
- Total Stops Num

### Response Variable
- **Price (continuous)** → converted to binary class labels:
  - **Class 0:** Cheap (Price ≤ Median)
  - **Class 1:** Expensive (Price > Median)

The median price (8,372 INR) results in a nearly balanced dataset (~50/50 split).

---

## 📉 Dimensionality Reduction (PCA)
- All predictors were standardized using z-score normalization
- PCA was applied to the engineered feature matrix
- **Optimal number of components:**  
  **m\* = 3**, retaining **≈93.5%** of total variance
- Reconstructed features were used as neural network inputs to preserve interpretability while reducing noise

---

## 🧠 Neural Network Models

### Baseline Architecture
⟨m\*, 8, 4, 1⟩  
- ReLU activations in hidden layers  
- Sigmoid output for probability prediction  
- Optimizer: Adam  
- Loss Function: Binary Cross-Entropy  

### Fine-Tuned Architecture
⟨m\*, 16, 8, 4, 1⟩  
- Increased depth and width
- Improved class-wise balance and validation accuracy
- More stable convergence behavior

---

## ⚙️ Experimental Setup
- **Train–Validation Split:** 70–30 (stratified)
- **Epochs:** 1000
- **Repeated Trials:** 20 independent random splits
- **Evaluation Metrics:**
  - Accuracy
  - Misclassification Error
  - Confusion Matrices
  - Class-wise Mean Squared Error (MSE)
  - Decision Boundary Visualization

---

## 📈 Key Findings
- PCA reconstruction preserves essential feature structure
- Duration Minutes is the most influential predictor
- Fine-tuned architecture improves validation accuracy and class balance
- Model performance is stable across repeated random splits
- Sensitivity analysis enhances interpretability by identifying dominant predictors
- Removing key features reduces separability and model effectiveness

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib

---

## 📂 Repository Contents
```
├──  Data_Train.csv
├── Final_Project_Predictive.ipynb
├── Final_Project_Predictive_Analytics_Report.pdf
└── README.md

```



## 👤 Author
**Saroar Jahan Shuba**  
Predictive Analytics  
December 2025

### 📎 Files
- [Dataset: Data_Train.csv](https://github.com/user-attachments/files/24349383/Data_Train.csv)
- [Jupyter Notebook](https://github.com/user-attachments/files/24349384/Final_Project_Predictive.ipynb)
- [Final Project Report (PDF)](https://github.com/user-attachments/files/24349385/Final_Project_Predictive_L20609025.pdf)


