# train.py
import os, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump


CSV_PATH = r"dataset/labeled_data.csv"  # your current path

# 1) Path + existence
print(f"[INFO] Using CSV at: {os.path.abspath(CSV_PATH)}")
if not os.path.exists(CSV_PATH):
    sys.exit(f"[ERROR] File not found: {CSV_PATH}")

# 2) Peek at file size (Kaggle file is usually a few MB)
print(f"[INFO] File size: {os.path.getsize(CSV_PATH)/1_000_000:.2f} MB")

# 3) Load and sanity-check columns
df = pd.read_csv(CSV_PATH, encoding="utf-8")
print("[INFO] Columns:", list(df.columns))

required = {"class","tweet"}
if not required.issubset(df.columns):
    # Some releases have "Unnamed: 0" instead of "id" — that’s fine.
    sys.exit(f"[ERROR] Required columns missing. Need at least {required}, got {set(df.columns)}")

# 4) Basic shape + preview
print(f"[INFO] Shape: {df.shape}")
print("[INFO] Head:\n", df[["class","tweet"]].head(3).to_string(index=False))

# 5) Label distribution (Kaggle Davidson dataset rough counts)
print("[INFO] class value counts:\n", df["class"].value_counts(dropna=False))

# Sanity expectations (not exact, but typical for Davidson et al.)
# total rows roughly ~ 24k–25k, class distribution: 1 (offensive) >> 2 (neither) >> 0 (hate)
n = len(df)
if not (20000 <= n <= 30000):
    print("[WARN] Row count atypical for the Kaggle dataset (~24–25k). Check source.")

def clean(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)         # strip URLs
    t = re.sub(r"[^a-z\s]", " ", t)       # keep letters/spaces
    return re.sub(r"\s+", " ", t).strip()

# 1) Load
df = pd.read_csv(CSV_PATH)

# Expect columns: id, count, hate_speech, offensive_language, neither, class, tweet
if "tweet" not in df.columns or "class" not in df.columns:
    raise ValueError(f"CSV missing required columns. Found: {list(df.columns)[:12]}")

# 2) Make binary label BEFORE using text
# class: 0=hate, 1=offensive, 2=neither -> binary {1 if hate/offensive else 0}
df["label"] = df["class"].apply(lambda x: 1 if x in (0, 1) else 0)

# 3) Create 'text' from 'tweet' and clean
df["text"] = df["tweet"].astype(str)
df["clean"] = df["text"].map(clean)

# Optional: drop empties
df = df[df["clean"].str.len() > 0]

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 5) Pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, stop_words="english")),
    ("clf", LogisticRegression(max_iter=300))
])

# 6) Train + eval
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("\nEvaluation report:")
print(classification_report(y_test, pred, digits=3))

# 7) Save model
dump(pipe, "model.joblib")
print("Model saved as model.joblib")
