import pickle
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, clean_data
from feature_engineering import create_features

df = load_data("../data/raw/insurance_claims.csv")
df = clean_data(df)
df = create_features(df)

y = df["total_claim_amount"]
X = df.drop(columns=["total_claim_amount","fraud_reported"], errors="ignore")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = pickle.load(open("../models/random_forest_model.pkl","rb"))

pred = model.predict(X_test)

print("R2:", r2_score(y_test,pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,pred)))