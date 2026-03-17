import pandas as pd

def create_features(df):

    df["claim_per_month"] = df["total_claim_amount"] / (df["months_as_customer"] + 1)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[18,30,40,50,60,100],
        labels=["18-30","30-40","40-50","50-60","60+"]
    )

    df = pd.get_dummies(df, drop_first=True)

    return df