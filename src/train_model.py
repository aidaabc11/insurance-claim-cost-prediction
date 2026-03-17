import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

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

param_grid = {
    "n_estimators":[200,300],
    "max_depth":[10,20,None]
}

rf = RandomForestRegressor()

grid = GridSearchCV(rf,param_grid,cv=5,n_jobs=-1)

grid.fit(X_train,y_train)

best_model = grid.best_estimator_

with open("../models/random_forest_model.pkl","wb") as f:
    pickle.dump(best_model,f)

print("Model trained and saved.")