# src/train.py

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- 1. Find the CSV (same idea as in inspect_data.py) ----------
BASE_DIR = Path(__file__).resolve().parents[1]

# Try normal "data" folder first
EXPECTED_NAME = "data"
data_dir = BASE_DIR / EXPECTED_NAME

# If that doesn't exist, accept anything that starts with "data" (like "data:")
if not data_dir.exists():
    candidates = [
        p for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("data")
    ]
    if not candidates:
        raise FileNotFoundError(f"No data folder found in {BASE_DIR}")
    data_dir = candidates[0]

# Try the expected file name
data_path = data_dir / "salary_data_cleaned.csv"

# If not found, search for something that looks like the salary CSV
if not data_path.exists():
    matches = list(data_dir.rglob("salary*cleaned*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find salary_data_cleaned.csv in {data_dir}"
        )
    data_path = matches[0]

print("Loading data from:", data_path)

df = pd.read_csv(data_path)

# ---------- 2. Choose target column ----------
# In the Glassdoor salary dataset this column is usually called "avg_salary".
TARGET_COL = "avg_salary"

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found. "
        f"Available columns: {list(df.columns)}"
    )

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ---------- 3. Split numeric / categorical features ----------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ---------- 4. Build preprocessing + model pipeline ----------
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# ---------- 5. Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining model...")
pipeline.fit(X_train, y_train)

# ---------- 6. Evaluate ----------
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.2f}")
print(f"R² : {r2:.3f}")

# ---------- 7. Save model ----------
MODEL_PATH = BASE_DIR / "models" / "salary_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

artifact = {
    "model": pipeline,
    "feature_names": X.columns.tolist(),
    "target_col": TARGET_COL,
}

joblib.dump(artifact, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
