# Supervised Classification Benchmark: Logistic Regression vs. Decision Tree

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/pandureddypandureddy798-collab/iris-classification-benchmark910)

This project implements an end-to-end supervised machine learning classification workflow on the Iris benchmark dataset according to all specified project requirements.

## Project Structure
- [`iris_classification.ipynb`](file:///c:/Users/phani/Documents/new%20intership/iris_classification.ipynb): Complete Jupyter Notebook containing data loading, Seaborn pairplot EDA, 80/20 train/test split, StandardScaler feature scaling, model training, metric benchmarking (accuracy, precision, recall, f1-score), confusion matrix heatmaps, and 2D decision boundary visualization.
- [`run_pipeline.py`](file:///c:/Users/phani/Documents/new%20intership/run_pipeline.py): Standalone Python script that executes the complete ML pipeline, prints benchmark tables, and exports high-resolution plot figures.

---

## Technical Approach & Implemented Requirements

| Requirement | Implementation Detail | Status |
| :--- | :--- | :---: |
| **Dataset Ingestion** | Loaded with Pandas from the benchmark URL with Scikit-Learn fallback | Complete |
| **Exploratory Data Analysis** | Pairplot using Seaborn (`sns.pairplot`) & correlation heatmaps | Complete |
| **Train/Test Splitting** | 80% Training / 20% Testing split with stratified class distributions (`random_state=42`) | Complete |
| **Feature Scaling** | Standardized using Scikit-Learn's `StandardScaler` (fitted on train only) | Complete |
| **Model Training** | `LogisticRegression(max_iter=200)` and `DecisionTreeClassifier(max_depth=3)` | Complete |
| **Performance Benchmarking** | Accuracy, Precision, Recall, and F1-Score (macro & weighted) + Classification Reports | Complete |
| **Visualizations** | Side-by-side Confusion Matrix heatmaps and 2D Decision Boundary surface plots | Complete |

---

## Benchmark Results Summary (Test Partition: 20%)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9667** | **0.9697** | **0.9667** | **0.9666** |
| **Decision Tree Classifier** | **0.9667** | **0.9697** | **0.9667** | **0.9666** |

---

## How to Run

### Option 1: Jupyter Notebook
Open [`iris_classification.ipynb`](file:///c:/Users/phani/Documents/new%20intership/iris_classification.ipynb) in Jupyter Notebook, VS Code, or Google Colab and run all cells sequentially.

### Option 2: Command Line Pipeline
Run the script using Python:
```bash
python run_pipeline.py
```
Generated visualization figures will be stored in the `plots/` directory.
