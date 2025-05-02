# src/train.py
import numpy as np
import os
import pandas as pd 
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuration ---
BASE_DIR = os.path.dirname(__file__) 
DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
MODEL_DIR = os.path.join(BASE_DIR, os.pardir, 'model')

# --- Use NEW dataset filename ---
DATASET_FILENAME = 'ml_host_training_data.csv' 
# --- Use NEW model output filename ---
MODEL_FILENAME = 'ml_host_pipeline.pkl' # New model for HostML

DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# ML Parameters
POLYNOMIAL_DEGREE = 2 
TEST_SIZE = 0.2         
RANDOM_STATE = 42       

# --- Load Data ---
print(f"Loading dataset for HostML training from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    print(f"--- ERROR: Dataset file not found: '{DATASET_PATH}' ---")
    print("Please run generate_dataset.py first to create the simplified data.")
    exit()
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"--- ERROR loading dataset: {e} ---")
    exit()

# --- Data Cleaning (If needed - unlikely for these simple features) ---
# df.dropna(inplace=True) 

# --- Feature Engineering & Selection ---
print("\nSelecting features and target for HostML model...")

# --- Define SIMPLIFIED feature columns - MUST match CSV header ---
feature_columns = [
    'cwnd_before_adj',
    'ack_received',
    'timeout_occurred',
    'loss_indicator',
]
try:
    X = df[feature_columns]
except KeyError as e:
    print(f"--- ERROR: Feature column missing from CSV: {e} ---")
    print(f"Available columns: {list(df.columns)}")
    exit()

# Define the target variable column
target_column = 'target_cwnd_after_adj' 
try:
    Y = df[target_column]
except KeyError as e:
     print(f"--- ERROR: Target column '{target_column}' missing from CSV ---")
     exit()

print(f"Selected {len(feature_columns)} features: {feature_columns}")
print(f"Target variable: '{target_column}'")
print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}")
if X.shape[0] == 0: 
    print("--- ERROR: No data left after feature selection ---"); exit()

# --- Train/Test Split ---
print(f"\nSplitting data (Test size: {TEST_SIZE:.0%})...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Model Pipeline (Same structure) ---
print(f"\nCreating model pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('poly', PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False, interaction_only=False)), 
    ('regressor', LinearRegression()) 
])
print(f"Pipeline steps: Scaler -> PolynomialFeatures(degree={POLYNOMIAL_DEGREE}) -> LinearRegression")

# --- Train Model (Trains on new simplified data) ---
print("\nTraining HostML model pipeline...")
try:
    pipeline.fit(X_train, Y_train)
    print("Training completed.")
except Exception as e:
     print(f"--- ERROR during model training: {e} ---"); exit()
     
# --- Evaluate Model ---
print("\nEvaluating HostML model performance...")
Y_train_pred = pipeline.predict(X_train); Y_test_pred = pipeline.predict(X_test)
train_r2 = r2_score(Y_train, Y_train_pred); test_r2 = r2_score(Y_test, Y_test_pred)
train_mae = mean_absolute_error(Y_train, Y_train_pred); test_mae = mean_absolute_error(Y_test, Y_test_pred)
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred)); test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
print(f"  Training Set: R2={train_r2:.4f}, MAE={train_mae:.4f}, RMSE={train_rmse:.4f}")
print(f"  Testing Set:  R2={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

# --- Example Test Predictions (Uses new simplified features) ---
print("\n--- Example Test Predictions (HostML Model) ---")
example_data = pd.DataFrame([
    {'cwnd_before_adj': 10, 'ack_received': 1, 'timeout_occurred': 0, 'loss_indicator': 0},
    {'cwnd_before_adj': 20, 'ack_received': 0, 'timeout_occurred': 1, 'loss_indicator': 1},
    {'cwnd_before_adj': 5, 'ack_received': 0, 'timeout_occurred': 0, 'loss_indicator': 0},
], columns=feature_columns) # Use the NEW feature_columns list

print("Sample Input Features:")
print(example_data)
example_predictions = pipeline.predict(example_data)
print("\nPredicted Target CWND:")
print(np.round(example_predictions, 2)) 

# --- Save Model Pipeline (Saves the NEWLY trained pipeline) ---
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"\nSaving trained HostML pipeline to: {MODEL_SAVE_PATH}")
try:
    with open(MODEL_SAVE_PATH, 'wb') as f: pickle.dump(pipeline, f)
    print("Pipeline saved successfully.")
except Exception as e: print(f"--- ERROR saving pipeline: {e} ---")

print("\nTraining script for HostML finished.")