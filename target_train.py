from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from target_extractor import build_target_feature


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required_columns = {"text", "name", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required_columns)}. Found: {list(df.columns)}")

    df = df[list(required_columns)].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[(df["text"] != "") & (df["name"] != "")]
    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")
    if set(df["label"].unique()) - {0, 1}:
        raise ValueError("Label column must contain only 0 or 1.")
    return df


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def can_validate(labels: pd.Series, test_size: float, total_samples: int) -> bool:
    if test_size <= 0 or total_samples < 10 or labels.nunique() < 2:
        return False
    test_count = max(int(round(total_samples * test_size)), 1)
    train_count = total_samples - test_count
    return test_count >= 2 and train_count >= 2 and int(labels.value_counts().min()) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight target-selector model.")
    parser.add_argument("--data", required=True, help="CSV file with columns: text,name,label")
    parser.add_argument("--output-dir", required=True, help="Directory to save the trained model.")
    parser.add_argument("--task-name", default="target", help="Saved model prefix.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.data))
    features = df.apply(lambda row: build_target_feature(row["text"], row["name"]), axis=1)
    labels = df["label"]
    model = build_model()

    metrics: dict[str, object] = {
        "task_name": args.task_name,
        "samples": int(len(df)),
        "positive_samples": int((labels == 1).sum()),
        "negative_samples": int((labels == 0).sum()),
    }

    if can_validate(labels, args.test_size, len(df)):
        x_train, x_valid, y_train, y_valid = train_test_split(
            features,
            labels,
            test_size=args.test_size,
            random_state=42,
            stratify=labels,
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics["validation_accuracy"] = accuracy_score(y_valid, predictions)
        metrics["classification_report"] = classification_report(
            y_valid,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    else:
        model.fit(features, labels)
        metrics["validation_accuracy"] = None
        metrics["classification_report"] = None

    model_path = output_dir / f"{args.task_name}_selector.joblib"
    metrics_path = output_dir / f"{args.task_name}_selector_metrics.json"
    metadata_path = output_dir / f"{args.task_name}_selector_metadata.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "task_name": args.task_name,
                "threshold": 0.5,
                "feature_format": "sentence + candidate_name + local_context",
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "joblib_version": joblib.__version__,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    if metrics["validation_accuracy"] is not None:
        print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
    else:
        print("Validation skipped.")


if __name__ == "__main__":
    main()
