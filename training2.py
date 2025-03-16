import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import joblib

# Generate synthetic data (replace with your dataset)
X, y = make_regression(n_samples=2000, n_features=31, noise=0.1, random_state=42)

# Convert to DataFrame for easier manipulation
X = pd.DataFrame(X)
X.columns = [f'feature_{i}' for i in range(1, X.shape[1] + 1)]  # Rename columns as feature_1, feature_2, ...

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Support Vector Regressor (SVR)
svr_model = SVR(C=10, gamma=0.1)
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)

# Save the SVR model
joblib.dump(svr_model, "svr_model.pkl")

# Calculate metrics for SVR
svr_mse = mean_squared_error(y_test, svr_pred)
svr_r2 = r2_score(y_test, svr_pred)
svr_mae = mean_absolute_error(y_test, svr_pred)
print(f"SVR Model -> MSE: {svr_mse:.4f}, R2: {svr_r2:.4f}, MAE: {svr_mae:.4f}")

# Scatter plot for SVR predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, svr_pred, alpha=0.6, color="blue", label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Ideal Fit")
plt.title("SVR: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.savefig("svr_predictions.png")
plt.show()

# Train the Support Vector Classifier (optional for classification tasks)
svc_model = SVC(kernel="rbf", C=10, gamma=0.1)  # Example with similar parameters
# Uncomment if your target `y` is categorical:
# svc_model.fit(X_train, y_train)

# Save the SVM/SVC model if used
joblib.dump(svc_model, "svc_model.pkl")
