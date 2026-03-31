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
from sklearn.svm import LinearSVC


def build_model(model_name: str) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
    )

    if model_name == "linear_svc":
        classifier = LinearSVC(C=1.0, dual="auto")
    elif model_name == "logreg":
        classifier = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )


def load_dataset(data_path: Path, text_col: str, label_col: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(data_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{text_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[(df[text_col] != "") & (df[label_col] != "")]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    return df[text_col], df[label_col]


def can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and int(counts.min()) >= 2


def should_run_validation(labels: pd.Series, test_size: float, total_samples: int) -> bool:
    if test_size <= 0 or total_samples < 5 or labels.nunique() <= 1:
        return False

    if test_size < 1:
        test_count = int(round(total_samples * test_size))
    else:
        test_count = int(test_size)

    test_count = max(test_count, 1)
    train_count = total_samples - test_count
    num_classes = int(labels.nunique())
    return test_count >= num_classes and train_count >= num_classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight text classifier.")
    parser.add_argument("--task-name", required=True, help="Task name, e.g. intent or target.")
    parser.add_argument("--data", required=True, help="CSV file path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the trained model.")
    parser.add_argument("--text-col", default="text", help="Text column name in CSV.")
    parser.add_argument("--label-col", default="label", help="Label column name in CSV.")
    parser.add_argument(
        "--model",
        default="linear_svc",
        choices=["linear_svc", "logreg"],
        help="Classifier type.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split ratio. Set 0 to train on all data without validation.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, labels = load_dataset(data_path, args.text_col, args.label_col)
    model = build_model(args.model)

    metrics: dict[str, object] = {
        "task_name": args.task_name,
        "model": args.model,
        "samples": int(len(texts)),
        "labels": sorted(labels.unique().tolist()),
    }

    if should_run_validation(labels, args.test_size, len(texts)):
        stratify = labels if can_stratify(labels) else None
        x_train, x_valid, y_train, y_valid = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=42,
            stratify=stratify,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        metrics["validation_accuracy"] = accuracy_score(y_valid, predictions)
        metrics["classification_report"] = classification_report(
            y_valid,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    else:
        model.fit(texts, labels)
        metrics["validation_accuracy"] = None
        metrics["classification_report"] = None

    model_path = output_dir / f"{args.task_name}_model.joblib"
    metrics_path = output_dir / f"{args.task_name}_metrics.json"
    metadata_path = output_dir / f"{args.task_name}_metadata.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "task_name": args.task_name,
                "text_col": args.text_col,
                "label_col": args.label_col,
                "model": args.model,
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
