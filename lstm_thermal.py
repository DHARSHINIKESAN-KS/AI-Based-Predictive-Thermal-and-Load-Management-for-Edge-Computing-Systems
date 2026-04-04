"""
Phase 2: LSTM Model for Predictive Thermal & Load Management
Step 1 — Load and normalize data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\edge_simulation_data.csv"
OUTPUT_DIR  = r"C:\Users\chan3\Documents\CLG\sensors\sp\data"

LOOK_BACK   = 30    # use last 30 seconds to make a prediction
LOOK_AHEAD  = 10    # predict 10 seconds into the future
TEST_SPLIT  = 0.2   # use last 20% of data for testing

# Features fed INTO the LSTM
FEATURE_COLS = ['cpu_load_pct', 'cpu_temp_c', 'gpu_temp_c', 'memory_pct', 'power_watts']

# What the LSTM will PREDICT
TARGET_COLS  = ['cpu_temp_c', 'cpu_load_pct']

# ─────────────────────────────────────────────
# STEP 1 — LOAD & NORMALIZE
# ─────────────────────────────────────────────
print("=" * 50)
print("  Phase 2: LSTM Thermal Prediction Model")
print("=" * 50)

# Load CSV
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
print(f"\n[1/4] Loaded {len(df)} samples from CSV")
print(f"      Columns: {list(df.columns)}")

# Keep only numeric feature columns
data = df[FEATURE_COLS].values  # shape: (3600, 5)

# Normalize everything to range [0, 1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Separate scaler just for targets (needed to inverse-transform predictions later)
target_scaler = MinMaxScaler()
target_scaler.fit(df[TARGET_COLS].values)

print(f"\n      Raw cpu_temp range : {df['cpu_temp_c'].min():.1f} – {df['cpu_temp_c'].max():.1f}°C")
print(f"      After scaling      : {data_scaled[:, 1].min():.3f} – {data_scaled[:, 1].max():.3f}")
print("\n✓ Step 1 complete — data loaded and normalized")




# ─────────────────────────────────────────────
# STEP 2 — CREATE SLIDING WINDOW SEQUENCES
# ─────────────────────────────────────────────

def create_sequences(data_scaled, df_targets, look_back, look_ahead):
    """
    Builds (X, y) pairs using a sliding window.
    X shape: (samples, look_back, n_features)  → 3D input for LSTM
    y shape: (samples, n_targets)               → what we want to predict
    """
    X, y = [], []

    for i in range(len(data_scaled) - look_back - look_ahead):
        # Input: 30 timesteps of all 5 features
        X.append(data_scaled[i : i + look_back])

        # Target: temp and load at t + look_ahead
        target_row = df_targets[i + look_back + look_ahead]
        y.append(target_row)

    return np.array(X), np.array(y)

# Get scaled target values
targets_scaled = target_scaler.transform(df[TARGET_COLS].values)

# Build sequences
X, y = create_sequences(data_scaled, targets_scaled, LOOK_BACK, LOOK_AHEAD)

# Train / test split — keep time order, don't shuffle
split_idx = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n[2/4] Sliding window sequences created")
print(f"      X shape      : {X.shape}  → (samples, timesteps, features)")
print(f"      y shape      : {y.shape}  → (samples, targets)")
print(f"      Training set : {X_train.shape[0]} samples")
print(f"      Test set     : {X_test.shape[0]} samples")
print("\n✓ Step 2 complete — sequences ready for LSTM")





# ─────────────────────────────────────────────
# STEP 3 — BUILD AND TRAIN THE LSTM
# ─────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

MODEL_PATH = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\lstm_thermal.keras"

# Build the model
model = Sequential([
    LSTM(64, input_shape=(LOOK_BACK, len(FEATURE_COLS)),
         return_sequences=True),        # passes sequence to next LSTM layer
    Dropout(0.2),                       # randomly drops 20% of neurons — prevents overfitting
    LSTM(32, return_sequences=False),   # final LSTM, outputs single vector
    Dropout(0.2),
    Dense(16, activation='relu'),       # small dense layer to refine prediction
    Dense(len(TARGET_COLS))             # output: 2 values (temp + load)
])

model.summary()

model.compile(
    optimizer='adam',
    loss='mse',                         # mean squared error — standard for regression
    metrics=['mae']                     # mean absolute error — easier to interpret
)

# Callbacks — these monitor training and stop early if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,           # stop if no improvement for 10 epochs
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,   # only saves when model improves
    verbose=0
)

print("\n[3/4] Training LSTM model...")
print("      This will take 1-3 minutes on CPU\n")

history = model.fit(
    X_train, y_train,
    epochs=100,             # max 100 epochs, early stopping will kick in sooner
    batch_size=32,
    validation_split=0.1,   # uses 10% of training data to monitor overfitting
    callbacks=[early_stop, checkpoint],
    verbose=1               # shows progress bar per epoch
)

print(f"\n✓ Step 3 complete — model trained and saved to {MODEL_PATH}")
print(f"  Stopped at epoch {len(history.history['loss'])}")
print(f"  Final training MAE   : {history.history['mae'][-1]:.4f}")
print(f"  Final validation MAE : {history.history['val_mae'][-1]:.4f}")



# ─────────────────────────────────────────────
# STEP 4 — EVALUATE AND PLOT RESULTS
# ─────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\n[4/4] Evaluating model on test set...")

# Generate predictions on test data
y_pred_scaled = model.predict(X_test, verbose=0)

# Convert predictions and actual values back to original scale (°C and %)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_actual = target_scaler.inverse_transform(y_test)

# Split into individual targets
temp_actual = y_actual[:, 0]
temp_pred   = y_pred[:, 0]
load_actual = y_actual[:, 1]
load_pred   = y_pred[:, 1]

# ── Metrics ──
temp_rmse = np.sqrt(mean_squared_error(temp_actual, temp_pred))
temp_mae  = mean_absolute_error(temp_actual, temp_pred)
load_rmse = np.sqrt(mean_squared_error(load_actual, load_pred))
load_mae  = mean_absolute_error(load_actual, load_pred)

print("\n── Model Performance on Test Set ────────────")
print(f"  CPU Temperature  → RMSE: {temp_rmse:.2f}°C  |  MAE: {temp_mae:.2f}°C")
print(f"  CPU Load         → RMSE: {load_rmse:.2f}%   |  MAE: {load_mae:.2f}%")

# ── Plot predicted vs actual ──
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("LSTM Prediction vs Actual — Test Set (10s ahead forecast)",
             fontsize=13, fontweight='bold')

# Plot only first 300 test samples so the chart is readable
n_show = 300
t = np.arange(n_show)

# Temperature plot
axes[0].plot(t, temp_actual[:n_show], label='Actual temp',
             color='#E74C3C', linewidth=1.2)
axes[0].plot(t, temp_pred[:n_show],   label='Predicted temp',
             color='#3498DB', linewidth=1.2, linestyle='--')
axes[0].axhline(75, color='#F39C12', linestyle=':', linewidth=1,
                label='Predictive target (75°C)')
axes[0].axhline(80, color='#C0392B', linestyle='--', linewidth=1,
                alpha=0.6, label='Reactive threshold (80°C)')
axes[0].fill_between(t,
    temp_actual[:n_show], temp_pred[:n_show],
    alpha=0.15, color='#3498DB', label='Prediction error')
axes[0].set_ylabel('Temperature (°C)', fontsize=10)
axes[0].set_title(f'CPU Temperature  —  RMSE: {temp_rmse:.2f}°C  |  MAE: {temp_mae:.2f}°C',
                  fontsize=10, loc='left')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(True, alpha=0.2)

# Load plot
axes[1].plot(t, load_actual[:n_show], label='Actual load',
             color='#27AE60', linewidth=1.2)
axes[1].plot(t, load_pred[:n_show],   label='Predicted load',
             color='#8E44AD', linewidth=1.2, linestyle='--')
axes[1].fill_between(t,
    load_actual[:n_show], load_pred[:n_show],
    alpha=0.15, color='#8E44AD', label='Prediction error')
axes[1].set_ylabel('CPU Load (%)', fontsize=10)
axes[1].set_xlabel('Time (seconds into test set)', fontsize=10)
axes[1].set_title(f'CPU Load  —  RMSE: {load_rmse:.2f}%  |  MAE: {load_mae:.2f}%',
                  fontsize=10, loc='left')
axes[1].legend(fontsize=8, loc='upper right')
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
eval_plot_path = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\lstm_evaluation.png"
plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
plt.close()

# ── Training history plot ──
fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history['loss'],     label='Training loss',   color='#E74C3C')
ax.plot(history.history['val_loss'], label='Validation loss', color='#3498DB')
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Loss (MSE)', fontsize=10)
ax.set_title('Training History — Loss Curve', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
loss_plot_path = r"C:\Users\chan3\Documents\CLG\sensors\sp\data\training_loss.png"
plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Evaluation plot → {eval_plot_path}")
print(f"  Loss curve plot → {loss_plot_path}")
print("\n✓ Phase 2 complete!")
print("  Your data folder now has everything needed for the paper.")