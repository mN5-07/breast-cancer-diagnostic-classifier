# Breast Cancer Wisconsin Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A beginner-friendly machine learning project that uses the **Breast Cancer Wisconsin (Diagnostic)** dataset to predict whether a breast tumor is **malignant** or **benign** using various scikit-learn classifiers.

This project was created as a learning exercise to master the full ML pipeline with scikit-learn: data exploration, preprocessing, model training, evaluation, and interpretability.

## Dataset
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Instances: 569
- Features: 30 real-valued features computed from digitized FNA images of breast masses
- Target: Diagnosis (M = malignant, B = benign)

## Project Goals
- Build and compare multiple classification models
- Achieve high accuracy, precision & recall (especially important for medical applications)
- Identify the most important features for malignancy prediction
- Learn scikit-learn best practices

## Technologies Used
- Python 3.x
- scikit-learn
- pandas & NumPy
- Matplotlib & Seaborn
- Jupyter Notebook

## Repository Structure
```
breast-cancer-wisconsin-classification/
├── data/                    # Raw & processed data (optional - usually not committed)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb     # Main exploration + modeling notebook
├── src/                     # (Optional) Reusable scripts/modules
├── models/                  # Saved models (optional)
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/breast-cancer-wisconsin-classification.git
   cd breast-cancer-wisconsin-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook notebooks/01_eda_and_modeling.ipynb
   ```

## Key Results (update after running!)
- Best model: [Random Forest / Logistic Regression / etc.]
- Test accuracy: XX.XX%
- Top features: worst concave points, worst radius, mean texture...

## Models Explored
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- (Optional: k-NN, Gradient Boosting, etc.)

## Insights
- Features describing cell shape & size irregularities (e.g., concave points, compactness) are strongest predictors of malignancy.
- Ensemble methods (Random Forest) consistently outperform simpler models.
- High recall is prioritized to minimize false negatives in a medical context.

## Future Work / Extensions
- Try deep learning with TensorFlow/Keras
- Feature selection techniques (RFE, SelectKBest)
- Deploy model as a simple web app (Streamlit/Flask)
- Add cross-validation & hyperparameter tuning with GridSearchCV

## Acknowledgments
- Dataset creators: Dr. William H. Wolberg, University of Wisconsin
- UCI Machine Learning Repository

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

