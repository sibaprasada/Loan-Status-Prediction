# 🏦 Loan Status Prediction System

This repository contains a Machine Learning project designed to predict whether a loan applicant will be approved or not based on various personal and financial details. The project utilizes a **Support Vector Machine (SVM)** model to classify the loan status.

## 📂 Dataset Details

The dataset contains historical data of loan applicants.
* **Total Rows:** 614
* **Total Columns:** 13 (before processing)
* **Key Features:** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`.
* **Target Variable:** `Loan_Status` (Y = Approved, N = Rejected).

## 🛠 Technologies Used

* **Python** (Core Language)
* **Pandas & NumPy** (Data Manipulation)
* **Seaborn & Matplotlib** (Data Visualization)
* **Scikit-Learn** (Model Building, Preprocessing, and Evaluation)

## ⚙️ Methodology

The project follows a structured Data Science pipeline:

### 1. Data Collection & Preprocessing
* **Handling Missing Values:**
    * Categorical columns (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`) were filled using the **Mode**.
    * Numerical columns (`LoanAmount`, `Loan_Amount_Term`) were filled using the **Median**.
* **Data Transformation:**
    * Replaced the value `'3+'` in the `Dependents` column with `4` to make it numerical.
* **Encoding Categorical Data:**
    * Converted categorical text data into numerical format using manual Label Encoding (e.g., Male: 0, Female: 1; Yes: 1, No: 0; Rural: 0, Semiurban: 1, Urban: 2).
* **Feature Selection:** Dropped the `Loan_ID` column as it does not contribute to the prediction.

### 2. Exploratory Data Analysis (EDA)
* Visualized the relationship between `Education` vs `Loan_Status` and `Married` vs `Loan_Status` using Count Plots to understand approval trends.

### 3. Data Splitting & Scaling
* Split the data into Training (90%) and Testing (10%) sets.
* Applied **StandardScaler** to normalize the feature values, ensuring that the model treats all input variables equally (crucial for SVM).

## 🤖 Model Used: Support Vector Machine (SVM)

I utilized the **Support Vector Classifier (SVC)** with a **Linear Kernel**.

### Why SVM?
1.  **Binary Classification:** The problem is a classic binary classification task (Yes/No), which SVM handles effectively by finding the optimal hyperplane to separate the two classes.
2.  **Accuracy:** SVM works well with higher-dimensional data (multiple features like Income, Credit History, etc.).
3.  **Robustness:** It is effective in cases where there is a clear margin of separation between classes.

## 📊 Model Performance

The model was evaluated using the Accuracy Score and Confusion Matrix.

| Metric | Score |
| :--- | :--- |
| **Training Data Accuracy** | **81.70%** |
| **Testing Data Accuracy** | **79.03%** |

* The Confusion Matrix on the test data showed good precision, particularly for positive loan approvals.

## 🔮 Predictive System

The notebook includes a custom predictive system. You can input the details of a new applicant manually (e.g., Income, Gender, Credit History), and the system will process the input (Encode -> Scale -> Predict) to output whether the loan is **Approved** or **Not Approved**.

## 🚀 How to Run

1.  Clone this repository.
2.  Install dependencies:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Loan_status_prediction.ipynb
    ```
4.  Run the cells to train the model and use the input system at the end to test specific cases.
