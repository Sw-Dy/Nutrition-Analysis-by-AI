# Machine Learning Model for Nutrition Density Prediction

## Overview
This repository contains a machine learning pipeline for predicting the nutrition density of food items. The pipeline includes data preprocessing, feature scaling, model training, hyperparameter tuning, and evaluation of multiple regression models, including:

- Random Forest
- XGBoost
- LightGBM
- Support Vector Regressor (SVR)
- Stacking Ensemble Model

Additionally, synthetic data generation and Support Vector Classification (SVC) models are included for further exploration.

## Dataset
The dataset used in this project should be provided as `FOOD-DATA-GROUP11.csv`. The dataset includes various nutritional features, such as:
- Fats, Carbohydrates, Sugars, Proteins
- Vitamins (A, B, C, D, E, K)
- Minerals (Calcium, Iron, Magnesium, etc.)
- Target variable: **Nutrition Density**

## Installation & Dependencies
To run this project, install the required dependencies using:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib
```

## Data Preprocessing
The preprocessing steps include:
1. Handling missing values using `SimpleImputer` (mean strategy)
2. Feature scaling using `MinMaxScaler`
3. Splitting data into training and testing sets

## Model Training & Evaluation
### 1. Model Selection
The following models are trained and evaluated:
- **Random Forest Regressor**
- **XGBoost Regressor**
- **LightGBM Regressor**
- **Support Vector Regressor (SVR)**
- **Stacking Ensemble Model**

### 2. Hyperparameter Tuning
Hyperparameter tuning is performed using `GridSearchCV` for the best parameter selection.

### 3. Evaluation Metrics
Each model is evaluated using:
- **Mean Squared Error (MSE)**
- **R-squared Score (R2)**
- **Mean Absolute Error (MAE)**

## Model Results
Each model's results are stored and visualized using matplotlib.
```bash
python main.py
```
- Feature importance is plotted for tree-based models.
- Model comparison graphs are generated.

## Saving & Loading Models
Trained models are saved using `joblib.dump()` and can be loaded using `joblib.load()`.
```python
import joblib
model = joblib.load("random_forest_model.pkl")
```

## Additional Components
- **Support Vector Classifier (SVC)** model is included for classification tasks.
- **Synthetic Data Generation** is used for testing ML models without real-world data.

## Repository Structure
```
├── README.md        # Project documentation
├── main.py          # Main script for running the pipeline
├── FOOD-DATA-GROUP11.csv  # Dataset
├── svr_model.pkl    # Saved Support Vector Regressor model
├── svc_model.pkl    # Saved Support Vector Classifier model
├── stacking_model.pkl  # Saved stacking ensemble model
└── svr_predictions.png # SVR prediction visualization
```

## Contributions
Feel free to contribute to this project by adding new models, optimizing existing ones, or improving visualizations.

## License
This project is open-source and available under the MIT License.
