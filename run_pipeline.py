"""
Supervised Classification Pipeline: Logistic Regression & Decision Tree Benchmark
Dataset: Iris Dataset
Author: Pair Programming Assistant
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def load_dataset() -> pd.DataFrame:
    """Load Iris dataset from CSV URL or Scikit-Learn fallback."""
    csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(csv_url)
        print("[INFO] Dataset successfully loaded from GitHub repository URL.")
    except Exception as err:
        print(f"[NOTICE] Direct URL load unavailable ({err}). Using Scikit-Learn fallback...")
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        name_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        if df["species"].dtype != object:
            df["species"] = df["species"].map(name_map)
    return df


def main():
    print("=" * 70)
    print("SUPERVISED CLASSIFICATION BENCHMARK: IRIS DATASET")
    print("=" * 70)

    # 1. Load Data
    df = load_dataset()
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    print(f"\n[INFO] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nClass Distribution:")
    print(df["species"].value_counts())

    # 2. Train / Test Split (80/20) with Stratification
    X = df[feature_cols].values
    y = df["species"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[INFO] Split complete: Train set={X_train.shape[0]} samples, Test set={X_test.shape[0]} samples")

    # 3. Feature Scaling with StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] StandardScaler applied (fitted on train set, applied to test set).")

    # 4. Train Logistic Regression & Decision Tree
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    print("[INFO] LogisticRegression and DecisionTreeClassifier trained successfully.")

    # 5. Predictions & Metrics
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_dt = dt_model.predict(X_test_scaled)

    def compute_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (Macro)": precision_score(y_true, y_pred, average="macro"),
            "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted"),
            "Recall (Macro)": recall_score(y_true, y_pred, average="macro"),
            "Recall (Weighted)": recall_score(y_true, y_pred, average="weighted"),
            "F1-Score (Macro)": f1_score(y_true, y_pred, average="macro"),
            "F1-Score (Weighted)": f1_score(y_true, y_pred, average="weighted"),
        }

    metrics_lr = compute_metrics(y_test, y_pred_lr)
    metrics_dt = compute_metrics(y_test, y_pred_dt)

    benchmark_table = pd.DataFrame([metrics_lr, metrics_dt], index=["Logistic Regression", "Decision Tree"]).round(4)
    print("\n" + "=" * 70)
    print("BENCHMARK METRICS SUMMARY TABLE")
    print("=" * 70)
    print(benchmark_table.to_string())

    print("\n" + "-" * 70)
    print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(y_test, y_pred_lr))

    print("-" * 70)
    print("DECISION TREE CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(y_test, y_pred_dt))

    # 6. Generate and Save Visualizations
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    labels = np.unique(y)

    # Confusion Matrix Plot
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"Logistic Regression (Accuracy: {metrics_lr['Accuracy']:.2f})")
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("Actual Class")

    sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"Decision Tree (Accuracy: {metrics_dt['Accuracy']:.2f})")
    axes[1].set_xlabel("Predicted Class")
    axes[1].set_ylabel("Actual Class")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[INFO] Confusion matrix heatmaps saved to '{cm_path}'.")

    # Decision Boundary Plot (Petal Length vs Petal Width)
    feat_indices = [2, 3]  # petal_length, petal_width
    X_2d_train = X_train[:, feat_indices]
    scaler_2d = StandardScaler()
    X_2d_train_sc = scaler_2d.fit_transform(X_2d_train)

    species_map = {name: idx for idx, name in enumerate(labels)}
    y_train_num = np.array([species_map[s] for s in y_train])

    lr_2d = LogisticRegression(max_iter=200, random_state=42).fit(X_2d_train_sc, y_train_num)
    dt_2d = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_2d_train_sc, y_train_num)

    x_min, x_max = X_2d_train_sc[:, 0].min() - 1, X_2d_train_sc[:, 0].max() + 1
    y_min, y_max = X_2d_train_sc[:, 1].min() - 1, X_2d_train_sc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(["#E6F0FA", "#FFF2E0", "#E8F5E9"])
    cmap_bold = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, model, title in [(axes[0], lr_2d, "Logistic Regression Boundary"), (axes[1], dt_2d, "Decision Tree Boundary")]:
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_light)
        ax.contour(xx, yy, Z, colors="gray", linewidths=0.7, linestyles="--")
        for idx, class_name in enumerate(labels):
            mask = (y_train_num == idx)
            ax.scatter(X_2d_train_sc[mask, 0], X_2d_train_sc[mask, 1],
                       c=cmap_bold[idx], label=f"{class_name}", edgecolor="k", s=40, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Standardized Petal Length")
        ax.set_ylabel("Standardized Petal Width")
        ax.legend(loc="upper left")

    plt.tight_layout()
    db_path = os.path.join(output_dir, "decision_boundaries.png")
    plt.savefig(db_path, dpi=300)
    plt.close()
    print(f"[INFO] Decision boundary visualization saved to '{db_path}'.")
    print("\n[SUCCESS] Pipeline completed successfully!")


if __name__ == "__main__":
    main()
