import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# -------------------- 1. Extract / Load --------------------
# Load dataset (replace with your file path or database query)
data = pd.read_csv("data.csv")

# Example: Separate features & target
X = data.drop("target", axis=1)
y = data["target"]

# -------------------- 2. Transform --------------------
# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Numeric pipeline: Impute missing values & scale
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical pipeline: Impute missing values & one-hot encode
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -------------------- 3. Load --------------------
# Create final pipeline (preprocessing + model, if desired)
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
    # You can add a model here, e.g. ("model", RandomForestClassifier())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit & transform training data
X_train_processed = full_pipeline.fit_transform(X_train)
X_test_processed = full_pipeline.transform(X_test)

# Optional: Save processed data to CSV
processed_df = pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed)
processed_df.to_csv("processed_data.csv", index=False
