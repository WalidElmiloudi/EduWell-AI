from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_FEATURES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate"
]

CATEGORICAL_FEATURES  = [
    "Gender",
    "Physical_Activity",
    "Diet_Quality",
    "Part_Time_Job",
    "Previous_Mental_Health_History"
]

TARGET_COLUMN = "Mental_Health_Risk"
DROP_COLUMNS = ["Student_ID"]


def get_preprocessor():
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def split_data(
    df,
    target_column = TARGET_COLUMN,
    test_size= 0.2,
    random_state = 42
):

    columns_to_drop = [col for col in DROP_COLUMNS + [target_column] if col in df.columns]
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor):
    feature_names = []
    # Numerical features stay as is
    feature_names.extend(NUMERICAL_FEATURES)
    # One-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_encoded_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names.extend(cat_encoded_cols)
    return feature_names

