import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import joblib

# Load Data
def load_data(calories_path, exercise_path):
    """
    Load and combine calories and exercise datasets.
    """
    calories = pd.read_csv(calories_path)
    exercise_data = pd.read_csv(exercise_path)
    calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
    return calories_data

# Preprocess Data
def preprocess_data(data):
    """
    Preprocess the data by replacing categorical values and splitting into features and target.
    """
    # Replace Gender with numeric values
    data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

    # Explicitly convert Gender column to integers to avoid downcasting warning
    data['Gender'] = data['Gender'].astype(int)
    
    # Features and target
    X = data.drop(columns=['User_ID', 'Calories'], axis=1)
    Y = data['Calories']
    return X, Y

# Train the Model
def train_model(X_train, Y_train):
    """
    Train an XGBoost regressor model.
    """
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    return model

# Evaluate the Model
def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model's performance and return evaluation metrics.
    """
    predictions = model.predict(X_test)
    mae = metrics.mean_absolute_error(Y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(Y_test, predictions))
    r2 = metrics.r2_score(Y_test, predictions)

    # Print metrics
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    return predictions

# Save the trained model to a file
def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Load the model from a file
def load_saved_model(filename):
    """
    Load a pre-trained model from a file.
    """
    return joblib.load(filename)

# Main function for training and saving the model
def train_and_save_model(calories_path, exercise_path, model_filename):
    """
    Train the model and save it.
    """
    # Load and preprocess data
    data = load_data(calories_path, exercise_path)
    X, Y = preprocess_data(data)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Train the model
    model = train_model(X_train, Y_train)

    # Evaluate the model
    evaluate_model(model, X_test, Y_test)

    # Save the trained model
    save_model(model, model_filename)

if __name__ == "__main__":
    # Define file paths and model save location
    calories_path = r'C:\Users\poova\Desktop\calories prediction\data\calories.csv'  # Modify this path
    exercise_path = r'C:\Users\poova\Desktop\calories prediction\data\exercise.csv'  # Modify this path
    model_filename = 'model.pkl'

    # Train and save the model
    train_and_save_model(calories_path, exercise_path, model_filename)
