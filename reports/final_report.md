# Breast Cancer Diagnosis Prediction

**Predicting malignant vs. benign breast tumors using machine learning on the classic Wisconsin Diagnostic Breast Cancer dataset.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project builds a complete machine learning pipeline to classify breast tumors as **malignant** (cancerous) or **benign** using features extracted from digitized fine needle aspirate (FNA) images of breast masses.

The dataset is the well-known **Breast Cancer Wisconsin (Diagnostic)** from UCI Machine Learning Repository — 569 samples, 30 real-valued features describing cell nuclei characteristics.

**Goal**: Achieve high predictive accuracy while understanding *which cellular features most strongly indicate malignancy*.

## Key Findings & Insights

- **Model Performance**: Top models reach **~96–97%** test accuracy and near-perfect AUC (~0.99).
- **Best Models**: Logistic Regression outperforms others in cross-validation and test metrics, with Random Forest and Linear SVM also performing strongly.
- **Most Important Features** (from permutation importance on the best model):
  1. **concavity_worst** — extreme depth/severity of concave portions in nucleus contour (strongest predictor)
  2. **texture_worst** — highest variation in gray-scale texture (surface roughness)
  3. **symmetry_worst** — greatest asymmetry in nucleus shape
  4. **concave points_worst** — most extreme number of indentations in contour
  5–10: Various mean and worst shape/size metrics (concave points_mean, concavity_mean, radius_worst, smoothness_worst, texture_se, area_worst)

These **"worst"** (most extreme) features dominate because malignant tumors exhibit greater irregularity, jaggedness, and heterogeneity in cell nuclei — aligning closely with known pathological indicators.

**Biological Takeaway**: The models learn the same visual cues pathologists use: chaotic nuclear shapes, asymmetry, and rough texture are strong malignancy signals.

## Project Structure

```
breast_cancer_prediction/
├── data/                    # raw (wdbc.data) + processed files
├── notebooks/               # Step-by-step Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_feature_importance_insights.ipynb
├── models/                  # Saved trained models (.joblib)
├── reports/                 # Final polished report
│   └── final_report.md      (or final_report.ipynb)
├── README.md
└── requirements.txt
```

## Tech Stack

- **Python** 3.8+
- **Core Libraries**:
  - pandas, numpy
  - scikit-learn (preprocessing, models, evaluation, interpretability)
  - matplotlib, seaborn (visualizations)
- **Environment**: Jupyter Notebook

## How to Run the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the raw dataset:
   - Place `wdbc.data` in the `data/` folder
   - Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

4. Explore the notebooks in order (01 → 04)

5. View the final insights:
   - `reports/final_report.md` (recommended)
   - Or browse the notebooks directly

## Results Summary Table

| Model              | Test Accuracy | CV Mean Accuracy (5-fold) | AUC (ROC) |
|--------------------|---------------|----------------------------|-----------|
| Logistic Regression| ~0.965        | ~0.971                     | ~0.996    |
| Random Forest      | ~0.974        | ~0.963                     | ~0.993    |
| SVM (Linear)       | ~0.965        | ~0.963                     | ~0.991    |
| Decision Tree      | ~0.921        | ~0.934                     | ~0.945    |

## Why Is the AUC So High?

The near-perfect AUC (~0.99) observed across multiple models is largely driven by the characteristics of the Wisconsin Diagnostic Breast Cancer dataset rather than model complexity alone.

The input features are expert-engineered measurements of cell nucleus morphology derived from digitized FNA images—properties that are already known to differ strongly between malignant and benign tumors. Features such as concavity, concave points, symmetry, and texture directly quantify the irregular and chaotic nuclear shapes that pathologists use in real clinical diagnosis.

In particular, the dataset includes “worst” (most extreme) feature values, which capture the most abnormal regions of a tumor. Malignancy often manifests in localized areas of extreme irregularity, making these worst-case measurements especially discriminative and leading to strong class separability.

This separability is further reflected in the strong performance of linear models such as Logistic Regression and Linear SVM, indicating that the malignant and benign classes are close to linearly separable in the feature space. Additionally, ROC/AUC is a threshold-independent metric that measures ranking performance; as long as malignant samples consistently receive higher predicted scores than benign ones, AUC values remain high.

While these results demonstrate effective learning of biologically meaningful patterns, it is important to note that this dataset is cleaner and more structured than many real-world clinical datasets. In practice, factors such as measurement noise, missing data, population shift, and imaging variability would likely reduce performance.


## Learning Outcomes

- Full ML pipeline: EDA → preprocessing → modeling → evaluation → interpretability
- Hands-on with scikit-learn: LabelEncoder, StandardScaler, train_test_split, GridSearchCV (optional), permutation_importance
- Importance of model interpretability in healthcare applications
- Understanding why certain features matter biologically

## Future Improvements

- Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
- Try boosting models (XGBoost, LightGBM)
- Feature selection experiments
- Deploy as a simple web app (Streamlit/Flask)

## Acknowledgments

- Dataset creators: Dr. William H. Wolberg, University of Wisconsin
- UCI Machine Learning Repository

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.