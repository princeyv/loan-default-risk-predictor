import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

st.title("💳 Loan Default Risk Predictor")
st.caption("Machine Learning model trained on the German Credit Dataset")

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳")

@st.cache_resource
def load_and_train_model():
    columns = [
        "checking_account_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_account",
        "employment_since",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "existing_credits",
        "job",
        "dependents",
        "telephone",
        "foreign_worker",
        "credit_risk"
    ]

    df = pd.read_csv(
        "data/german.data",
        sep=" ",
        header=None,
        names=columns
    )

    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    return model

model = load_and_train_model()

st.write("Enter applicant details to estimate default risk.")

checking_account_status = st.selectbox(
    "Checking Account Status",
    ["A11", "A12", "A13", "A14"]
)

duration = st.slider("Loan Duration (months)", 4, 72, 24)

credit_history = st.selectbox(
    "Credit History",
    ["A30", "A31", "A32", "A33", "A34"]
)

purpose = st.selectbox(
    "Purpose",
    ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"]
)

credit_amount = st.number_input("Credit Amount", min_value=250, max_value=20000, value=5000, step=100)

savings_account = st.selectbox(
    "Savings Account",
    ["A61", "A62", "A63", "A64", "A65"]
)

employment_since = st.selectbox(
    "Employment Since",
    ["A71", "A72", "A73", "A74", "A75"]
)

installment_rate = st.slider("Installment Rate", 1, 4, 2)

personal_status_sex = st.selectbox(
    "Personal Status / Sex",
    ["A91", "A92", "A93", "A94", "A95"]
)

other_debtors = st.selectbox(
    "Other Debtors",
    ["A101", "A102", "A103"]
)

residence_since = st.slider("Residence Since", 1, 4, 2)

property_val = st.selectbox(
    "Property",
    ["A121", "A122", "A123", "A124"]
)

age = st.slider("Age", 18, 75, 30)

other_installment_plans = st.selectbox(
    "Other Installment Plans",
    ["A141", "A142", "A143"]
)

housing = st.selectbox(
    "Housing",
    ["A151", "A152", "A153"]
)

existing_credits = st.slider("Existing Credits", 1, 4, 1)

job = st.selectbox(
    "Job",
    ["A171", "A172", "A173", "A174"]
)

dependents = st.slider("Dependents", 1, 2, 1)

telephone = st.selectbox(
    "Telephone",
    ["A191", "A192"]
)

foreign_worker = st.selectbox(
    "Foreign Worker",
    ["A201", "A202"]
)

input_data = pd.DataFrame([{
    "checking_account_status": checking_account_status,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "credit_amount": credit_amount,
    "savings_account": savings_account,
    "employment_since": employment_since,
    "installment_rate": installment_rate,
    "personal_status_sex": personal_status_sex,
    "other_debtors": other_debtors,
    "residence_since": residence_since,
    "property": property_val,
    "age": age,
    "other_installment_plans": other_installment_plans,
    "housing": housing,
    "existing_credits": existing_credits,
    "job": job,
    "dependents": dependents,
    "telephone": telephone,
    "foreign_worker": foreign_worker
}])

if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Default probability: **{probability:.2%}**")

    if prediction == 1:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk of Default")

    if probability > 0.7:
        st.warning("Applicant is very likely to default.")
    elif probability > 0.4:
        st.info("Applicant has moderate default risk.")
    else:
        st.success("Applicant appears financially safe.")
        
    st.markdown("---")
st.subheader("Model Information")

st.write("""
Model: Logistic Regression  
Dataset: German Credit Dataset (1000 applicants)  
Features: 20 financial attributes  
Evaluation ROC-AUC: ~0.80
""")