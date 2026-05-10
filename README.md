# 🚀 AI Financial Fraud Detection & Risk Scoring System

## 📌 Overview
This project develops a machine learning–based system for detecting fraudulent financial transactions and classifying them into actionable risk levels. It integrates data science, financial analysis, and audit principles to enhance fraud prevention, improve risk assessment, and support data-driven financial decision-making.

The system addresses real-world challenges such as extreme class imbalance in financial datasets and demonstrates how AI can improve transparency, strengthen internal controls, and support more resilient financial systems.

---

## 🎯 Key Features
- Fraud detection using machine learning models  
- Comparison of multiple models (Logistic Regression, Balanced Model, Random Forest)  
- Handling of highly imbalanced financial datasets  
- Risk scoring system (Low, Medium, High) based on fraud probability  
- Performance evaluation using precision, recall, and F1-score  

---

## 📊 Final Model Performance (Random Forest)
- Precision: **0.97**  
- Recall: **0.77**  
- F1-score: **0.86**  

The model achieves a strong balance between detecting fraudulent transactions and minimizing false positives, which is critical in financial systems.

---

## 📊 Visual Insights

### Confusion Matrix
![Confusion Matrix](Screenshots/confusion_matrix_random_forest.png)

### Fraud Distribution (%)
![Fraud Distribution](Screenshots/fraud_vs_normal_percentage.png)

### Transaction Distribution (Log Scale)
![Transaction Distribution](Screenshots/transaction_amount_distribution_log.png)

### Transaction Amount by Class (Outliers Removed)
![Boxplot](Screenshots/transaction_amount_by_class_boxplot.png)

---

## 🧠 Key Insight
Fraud detection is not just about accuracy. Due to severe class imbalance in financial datasets, models must balance:

- Detecting fraudulent transactions (**recall**)  
- Avoiding false alarms (**precision**)  

This project highlights the importance of probability-based decision-making and threshold tuning in real-world fraud detection systems.

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## 📁 Project Structure
```
AI-Financial-Fraud-Risk-Detection/
│
├── fraud_detection.py                # Main ML pipeline
├── README.md                         # Project documentation
├── .gitignore                        # Ignore large files
├── Screenshots/                      # Visual outputs
├── Outputs/                          # Prediction outputs
└── data/                             # Dataset (not uploaded)
```

---

## ▶️ How to Run

### 1. Clone the repository
```
git clone https://github.com/ttaanisa/AI-Financial-Fraud-Risk-Detection.git
```

### 2. Navigate into the project
```
cd AI-Financial-Fraud-Risk-Detection
```

### 3. Install dependencies
```
pip install pandas scikit-learn matplotlib
```

### 4. Run the script
```
python fraud_detection.py
```

---

## 📌 Note on Dataset
The dataset is not included due to GitHub file size limitations.

Download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Then place it in:
```
data/creditcard.csv
```

---

## 💡 Future Improvements
- Real-time fraud detection system (live transaction scoring)  
- Deployment using Streamlit, Flask, or FastAPI  
- Interactive dashboard for monitoring fraud risk  
- Integration with financial transaction systems  
- Advanced models (XGBoost, Neural Networks)  
- Model explainability (SHAP, LIME)  

---

## 🚀 Project Vision
This project serves as a foundation for building intelligent financial risk systems that can support banks, fintech platforms, and regulatory environments in detecting fraud more effectively and proactively.

---

## 👤 Author
**Takudzwa Taanisa**  
Master’s in Global Management (Data Science)  
Arizona State University  
