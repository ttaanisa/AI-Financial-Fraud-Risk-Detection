# 🚀 AI Financial Fraud Detection & Risk Scoring System

## 📌 Overview
This project develops a machine learning–based system for detecting fraudulent financial transactions and classifying them into actionable risk levels. It combines data science, financial analysis, and audit principles to support fraud prevention and financial decision-making.

---

## 🎯 Key Features
- Fraud detection using machine learning models  
- Comparison of multiple models (Logistic Regression, Balanced Model, Random Forest)  
- Handling of imbalanced financial datasets  
- Risk scoring system (Low, Medium, High)  
- Performance evaluation using precision, recall, and F1-score  

---

## 📊 Final Model Performance (Random Forest)
- Precision: **0.97**
- Recall: **0.77**
- F1-score: **0.86**

The model achieves a strong balance between detecting fraud and minimizing false positives.

---

## 🧠 Key Insight
Fraud detection is not just about accuracy. Due to severe class imbalance, models must balance:
- Detecting fraudulent transactions (recall)
- Avoiding false alarms (precision)

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
├── fraud_detection.py      # Main script
├── README.md               # Project documentation
├── .gitignore              # Ignore large files
└── data/                   # Dataset (not uploaded to GitHub)
```

---

## ▶️ How to Run

1. Clone the repository:
```
git clone https://github.com/ttaanisa/AI-Financial-Fraud-Risk-Detection.git
```

2. Navigate to the project folder:
```
cd AI-Financial-Fraud-Risk-Detection
```

3. Install dependencies:
```
pip install pandas scikit-learn matplotlib
```

4. Run the script:
```
python fraud_detection.py
```

---

## 📌 Note on Dataset

The dataset used for this project is not included in this repository due to GitHub file size limits.

You can download the dataset from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file in:
```
data/creditcard.csv
```

---

## 💡 Future Improvements
- Real-time fraud detection system  
- Deployment using Flask or FastAPI  
- Dashboard for monitoring fraud risk  
- Integration with financial transaction systems  

---

## 👤 Author
Takudzwa Taanisa  
Master’s in Global Management (Data Science) – Arizona State University  
