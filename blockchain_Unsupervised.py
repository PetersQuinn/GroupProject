# =====================================================
# Unsupervised Fraud / Outlier Detection via Autoencoder
# + Outlier thresholding, CSV export, and extra diagnostics
# =====================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------
# 1. Load dataset
# -----------------------
print("[1/8] Loading cleaned transfers parquet...")
df = pd.read_parquet("transfers_verified_small.parquet")
print("Shape:", df.shape)

# -----------------------
# 2. Feature engineering (address-level)
# -----------------------
print("[2/8] Building address-level features...")

def make_address_features(df):
    addr_in = df.groupby("to")["value"].agg(["count", "sum", "mean"]).rename(
        columns={"count": "n_in", "sum": "v_in_sum", "mean": "v_in_mean"}
    )
    addr_out = df.groupby("from")["value"].agg(["count", "sum", "mean"]).rename(
        columns={"count": "n_out", "sum": "v_out_sum", "mean": "v_out_mean"}
    )
    feat = addr_in.join(addr_out, how="outer").fillna(0)
    feat["net_flow"] = feat["v_in_sum"] - feat["v_out_sum"]
    feat["activity"] = feat["n_in"] + feat["n_out"]
    feat["avg_tx_value"] = (feat["v_in_mean"] + feat["v_out_mean"]) / 2
    return feat

addr_features = make_address_features(df)
print("Address feature shape:", addr_features.shape)

# -----------------------
# 3. Train / val / test split
# -----------------------
print("[3/8] Splitting train/val/test...")
X_train, X_temp = train_test_split(addr_features, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# -----------------------
# 4. Scaling
# -----------------------
print("[4/8] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# -----------------------
# 5. Build Autoencoder
# -----------------------
print("[5/8] Building Autoencoder model...")

input_dim = X_train_scaled.shape[1]
encoding_dim = input_dim // 2  # simple compression factor

def build_autoencoder(input_dim, encoding_dim):
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
    encoded = layers.Dense(encoding_dim // 2, activation="relu")(encoded)
    decoded = layers.Dense(encoding_dim, activation="relu")(encoded)
    output_layer = layers.Dense(input_dim, activation="linear")(decoded)
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

autoencoder = build_autoencoder(input_dim, encoding_dim)
autoencoder.summary()

# -----------------------
# 6. Train model
# -----------------------
print("[6/8] Training Autoencoder...")

es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(X_val_scaled, X_val_scaled),
    callbacks=[es],
    verbose=1,
)

# -----------------------
# 7. Reconstruction error + diagnostics
# -----------------------
print("[7/8] Computing reconstruction error and diagnostics...")

X_test_pred = autoencoder.predict(X_test_scaled)
mse = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

# Basic distribution diagnostics
percentiles = np.percentile(mse, [50, 90, 95, 99, 99.5, 99.9, 99.99])
print("Reconstruction error percentiles (50,90,95,99,99.5,99.9,99.99):")
print(percentiles)

# Address-level score table
addr_scores = pd.DataFrame(
    {
        "address": X_test.index,
        "reconstruction_error": mse,
    }
).sort_values("reconstruction_error", ascending=False)

print("\nTop 10 by reconstruction error:")
print(addr_scores.head(10))

# Full histogram
plt.figure(figsize=(8, 5))
plt.hist(np.log1p(mse), bins=100, alpha=0.8)
plt.title("Distribution of Reconstruction Error (log1p)")
plt.xlabel("log(1 + MSE)")
plt.ylabel("Count")
plt.savefig("ae_recon_error_hist_full.png", bbox_inches="tight")
plt.close()

# Zoomed histogram (optional, for slide clarity)
plt.figure(figsize=(8, 5))
plt.hist(np.log1p(mse), bins=100, range=(0, 1), alpha=0.8)
plt.title("Distribution of Reconstruction Error (log1p) – zoomed")
plt.xlabel("log(1 + MSE)")
plt.ylabel("Count")
plt.savefig("ae_recon_error_hist_zoom.png", bbox_inches="tight")
plt.close()

# -----------------------
# 8. Thresholding + CSV export
# -----------------------
print("[8/8] Thresholding and saving scores...")

# Example: mark top 0.5% as outliers
PERC = 99.5
threshold = np.percentile(mse, PERC)
print(f"Using threshold at {PERC}th percentile: {threshold:.4f}")

addr_scores["is_outlier"] = addr_scores["reconstruction_error"] >= threshold
print("\nOutlier flag counts:")
print(addr_scores["is_outlier"].value_counts())

# Save for downstream analysis (e.g., joining back to transfers)
addr_scores.to_csv("address_ae_scores.csv", index=False)
print("Saved address scores to address_ae_scores.csv")

print("\nTop 10 potential outliers (highest reconstruction error):")
print(addr_scores.head(10))
