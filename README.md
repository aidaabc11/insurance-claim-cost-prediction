
# Insurance Claim Cost Prediction & Fraud Detection

End‑to‑end data science project demonstrating a production‑style machine learning workflow for insurance analytics.

## Project Overview
This project analyzes insurance claim records to:

• Predict claim cost using regression models  
• Detect fraudulent claims using classification models  
• Identify customer segments using clustering  

The workflow reflects a real industry data science pipeline including:

- Data cleaning
- Feature engineering
- Exploratory data analysis
- Model comparison
- Hyperparameter tuning
- Model explainability
- Model persistence

## Tech Stack
Python, Pandas, NumPy, Scikit‑learn, Seaborn, Matplotlib

## Project Structure

data/  
 raw/ – original dataset  
 processed/ – cleaned datasets  

notebooks/  
 analysis notebook  

src/  
 data_preprocessing.py  
 feature_engineering.py  
 train_model.py  
 evaluate_model.py  

models/  
 saved trained models  

reports/  
 figures and visualizations  

## How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run training

python src/train_model.py

3. Evaluate model

python src/evaluate_model.py

## Author
Marzieh Abbasi
