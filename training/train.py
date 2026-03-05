from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib


def main():
    # 1) Скачиваем датасет BANKING77 с HuggingFace
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)

    # 2) Печатаем, какие части (splits) есть в датасете: train/test и т.д.
    print("Splits:", list(ds.keys()))

    # 3) Печатаем, какие колонки есть в train (например: text, label)
    print("Train columns:", ds["train"].column_names)

    # 4) Печатаем пример одной строки (чтобы увидеть формат)
    print("Example row:", ds["train"][0])

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(ds)

    print(f"Sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_val_pred = pipe.predict(X_val)
    evaluate(y_val, y_val_pred, name="val")

    y_test_pred = pipe.predict(X_test)

    evaluate(y_test, y_test_pred, name="test")

    save_model(pipe, "models/intent_clf.joblib")

  

def split_data(ds, val_size=0.2, random_state=42):
    X= ds['train']['text']
    y = ds['train']['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )
    X_test = ds["test"]["text"]
    y_test = ds["test"]["label"]

    return X_train, X_val, X_test, y_train, y_val, y_test



def build_pipeline(random_state=42):
    """
    Pipeline: TF-IDF -> LogisticRegression
    """
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ('model', LogisticRegression(max_iter=1000, random_state=random_state))
    ])   # TODO: Pipeline([...])
    return pipe

def evaluate(y_true, y_pred, name: str):
    acc = accuracy_score(y_true, y_pred)  # TODO: accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")   # TODO: f1_score(y_true, y_pred, average="macro")
    print(f"{name}: accuracy={acc:.4f} macro_f1={f1:.4f}")

def save_model(model, path: str):
    """
    Сохраняет обученный пайплайн (vectorizer + model) в файл.
    """
    joblib.dump(model, path)
    print(f"Saved model to: {path}")

if __name__ == "__main__":
    main()