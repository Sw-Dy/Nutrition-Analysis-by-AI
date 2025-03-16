import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
import joblib

# 1. Data Preprocessing
def preprocess_data(data, nutrient_columns):
    # Imputation for missing values
    imputer = SimpleImputer(strategy='mean')
    data[nutrient_columns] = imputer.fit_transform(data[nutrient_columns])

    # Feature Scaling
    scaler = MinMaxScaler()
    data[nutrient_columns] = scaler.fit_transform(data[nutrient_columns])

    return data, scaler, imputer

# Load the dataset
data = pd.read_csv('FOOD-DATA-GROUP11.csv')  # Ensure the file path is correct
nutrient_columns = [
    "Fat", "Saturated Fats", "Monounsaturated Fats", "Polyunsaturated Fats",
    "Carbohydrates", "Sugars", "Protein", "Dietary Fiber", "Cholesterol",
    "Sodium", "Vitamin A", "Vitamin B1", "Vitamin B11", "Vitamin B12",
    "Vitamin B2", "Vitamin B3", "Vitamin B5", "Vitamin B6", "Vitamin C",
    "Vitamin D", "Vitamin E", "Vitamin K", "Calcium", "Copper", "Iron",
    "Magnesium", "Manganese", "Phosphorus", "Potassium", "Selenium", "Zinc"
]

# Preprocess the data
data, scaler, imputer = preprocess_data(data, nutrient_columns)

# Target variable: Nutrition Density
X = data[nutrient_columns]
y = data['Nutrition Density']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Definitions
def create_model():
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=100),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(C=100, gamma=0.1, epsilon=0.1)
    }
    return models

# 3. Hyperparameter Optimization
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=1, 
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

# 4. Model Evaluation & Visualization
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"{model_name} -> MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
    
    # Plot Feature Importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        plt.barh(nutrient_columns, feature_importance)
        plt.title(f'{model_name} - Feature Importance')
        plt.show()

    return mse, r2, mae

# 5. Train and Evaluate Multiple Models
def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            # Define hyperparameter grid for tuning (can be customized for each model)
            param_grid = {
                "Random Forest": {"n_estimators": [50, 100]},
                "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                "LightGBM": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]},
                "SVR": {"C": [1, 10], "gamma": [0.01, 0.1]}
            }.get(name, {})

            best_params, best_model = hyperparameter_tuning(model, param_grid, X_train, y_train)
            print(f"Best Parameters for {name}: {best_params}")

            # Model Evaluation
            mse, r2, mae = evaluate_model(best_model, X_train, y_train, X_test, y_test, name)
            results[name] = {"MSE": mse, "R2": r2, "MAE": mae}

            # Save the best model
            joblib.dump(best_model, f"{name.lower().replace(' ', '_')}_model.pkl")
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    return results

# 6. Stacking Ensemble Model
def stacking_ensemble(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import StackingRegressor
    base_learners = [
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
        ('lgbm', LGBMRegressor(n_estimators=100, random_state=42))
    ]
    meta_learner = Ridge(alpha=1.0)

    stack_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)
    stack_model.fit(X_train, y_train)
    predictions = stack_model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Stacking Model -> MSE: {mse:.4f}, R2: {r2:.4f}")
    joblib.dump(stack_model, "stacking_model.pkl")

# 7. Visualizing Results
def plot_model_performance(results):
    models = list(results.keys())
    mse = [results[model]["MSE"] for model in models]
    r2 = [results[model]["R2"] for model in models]
    
    x = np.arange(len(models))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.bar(x - 0.2, mse, 0.4, label="MSE", color="blue")
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MSE', color="blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.tick_params(axis='y', labelcolor="blue")
    
    ax2 = ax1.twinx()
    ax2.plot(x, r2, label="R2", color="red", marker='o')
    ax2.set_ylabel('R2', color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    
    plt.title('Model Comparison')
    plt.tight_layout()
    plt.show()

# 8. Main Execution
models = create_model()
results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

# Stacking Model
stacking_ensemble(X_train, y_train, X_test, y_test)

# Plot the model comparison
plot_model_performance(results)
