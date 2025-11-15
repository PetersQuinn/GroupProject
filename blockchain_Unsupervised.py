import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("[1/10] Loading cleaned transfers parquet...")
df = pd.read_parquet("transfers_verified_small.parquet")
print("Shape:", df.shape)

print("[2/10] Building address-level features...")

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


print("[3/10] Splitting train/val/test...")
X_train, X_temp = train_test_split(addr_features, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

print("[4/10] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


print("[5/10] Building Autoencoder model...")

input_dim = X_train_scaled.shape[1]
encoding_dim = input_dim // 2  

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

print("[6/10] Training Autoencoder...")

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

print("[7/10] Computing reconstruction error and diagnostics...")

X_test_pred = autoencoder.predict(X_test_scaled)
mse = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)

percentiles = np.percentile(mse, [50, 90, 95, 99, 99.5, 99.9, 99.99])
print("Reconstruction error percentiles (50,90,95,99,99.5,99.9,99.99):")
print(percentiles)

addr_scores_ae = pd.DataFrame(
    {
        "address": X_test.index,
        "ae_recon_error": mse,
    }
).sort_values("ae_recon_error", ascending=False)

print("\nTop 10 by AE reconstruction error:")
print(addr_scores_ae.head(10))

plt.figure(figsize=(8, 5))
plt.hist(np.log1p(mse), bins=100, alpha=0.8)
plt.title("AE Reconstruction Error (log1p) – full")
plt.xlabel("log(1 + MSE)")
plt.ylabel("Count")
plt.savefig("ae_recon_error_hist_full.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(np.log1p(mse), bins=100, range=(0, 1), alpha=0.8)
plt.title("AE Reconstruction Error (log1p) – zoomed")
plt.xlabel("log(1 + MSE)")
plt.ylabel("Count")
plt.savefig("ae_recon_error_hist_zoom.png", bbox_inches="tight")
plt.close()

print("[8/10] Thresholding AE scores and saving...")

PERC = 99.5
ae_threshold = np.percentile(mse, PERC)
print(f"Using AE threshold at {PERC}th percentile: {ae_threshold:.4f}")

addr_scores_ae["ae_is_outlier"] = addr_scores_ae["ae_recon_error"] >= ae_threshold
print("\nAE outlier flag counts:")
print(addr_scores_ae["ae_is_outlier"].value_counts())

addr_scores_ae.to_csv("address_ae_scores.csv", index=False)
print("Saved AE scores to address_ae_scores.csv")

print("\nTop 10 AE potential outliers (highest reconstruction error):")
print(addr_scores_ae.head(10))

print("\n[9/10] Fitting Isolation Forest...")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.005,   
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X_train_scaled)


if_scores = -iso_forest.decision_function(X_test_scaled)  
if_labels = iso_forest.predict(X_test_scaled)  

addr_scores_if = pd.DataFrame(
    {
        "address": X_test.index,
        "if_anomaly_score": if_scores,
        "if_is_outlier": (if_labels == -1),
    }
).set_index("address")

print("Isolation Forest outlier counts:")
print(addr_scores_if["if_is_outlier"].value_counts())

print("\nTop 10 by Isolation Forest anomaly score:")
print(
    addr_scores_if.sort_values("if_anomaly_score", ascending=False)
    .head(10)
)

print("\n[10/10] Fitting KMeans and computing distance-based anomaly scores...")

N_CLUSTERS = 8
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

test_clusters = kmeans.predict(X_test_scaled)
centroids = kmeans.cluster_centers_

dists = np.linalg.norm(X_test_scaled - centroids[test_clusters], axis=1)

km_threshold = np.percentile(dists, PERC)  
km_is_outlier = dists >= km_threshold

addr_scores_km = pd.DataFrame(
    {
        "address": X_test.index,
        "km_cluster": test_clusters,
        "km_dist_to_centroid": dists,
        "km_is_outlier": km_is_outlier,
    }
).set_index("address")

print("KMeans distance-based outlier counts:")
print(addr_scores_km["km_is_outlier"].value_counts())

print("\nTop 10 by KMeans distance to centroid:")
print(
    addr_scores_km.sort_values("km_dist_to_centroid", ascending=False)
    .head(10)
)

print("\n[extra] Fitting Local Outlier Factor and computing anomaly scores...")

lof = LocalOutlierFactor(
    n_neighbors=20,
    novelty=True
)
lof.fit(X_train_scaled)

lof_scores = -lof.score_samples(X_test_scaled)
lof_threshold = np.percentile(lof_scores, PERC)
lof_is_outlier = lof_scores >= lof_threshold

addr_scores_lof = pd.DataFrame(
    {
        "address": X_test.index,
        "lof_score": lof_scores,
        "lof_is_outlier": lof_is_outlier,
    }
).set_index("address")

print("LOF outlier flag counts:")
print(addr_scores_lof["lof_is_outlier"].value_counts())

print("\nTop 10 by LOF anomaly score:")
print(
    addr_scores_lof.sort_values("lof_score", ascending=False)
    .head(10)
)

addr_scores_lof.reset_index().to_csv("address_lof_scores.csv", index=False)
print("Saved LOF scores to address_lof_scores.csv")

combined = (
    addr_scores_ae.set_index("address")
    .join(addr_scores_if, how="left")
    .join(addr_scores_km, how="left")
    .join(addr_scores_lof, how="left")
)

combined.reset_index().to_csv("address_unsupervised_scores_all.csv", index=False)
print("\nSaved combined unsupervised scores to address_unsupervised_scores_all.csv")

print("\nMethod agreement on test addresses:")
print("AE outliers:", combined["ae_is_outlier"].sum())
print("IF outliers:", combined["if_is_outlier"].sum())
print("KM outliers:", combined["km_is_outlier"].sum())
print("LOF outliers:", combined["lof_is_outlier"].sum())

both_ae_if = (combined["ae_is_outlier"] & combined["if_is_outlier"]).sum()
both_ae_km = (combined["ae_is_outlier"] & combined["km_is_outlier"]).sum()
both_if_km = (combined["if_is_outlier"] & combined["km_is_outlier"]).sum()
both_ae_lof = (combined["ae_is_outlier"] & combined["lof_is_outlier"]).sum()
both_if_lof = (combined["if_is_outlier"] & combined["lof_is_outlier"]).sum()
both_km_lof = (combined["km_is_outlier"] & combined["lof_is_outlier"]).sum()
all_three = (
    combined["ae_is_outlier"]
    & combined["if_is_outlier"]
    & combined["km_is_outlier"]
).sum()
all_four = (
    combined["ae_is_outlier"]
    & combined["if_is_outlier"]
    & combined["km_is_outlier"]
    & combined["lof_is_outlier"]
).sum()

print(f"Overlap AE ∩ IF: {both_ae_if}")
print(f"Overlap AE ∩ KM: {both_ae_km}")
print(f"Overlap IF ∩ KM: {both_if_km}")
print(f"Overlap AE ∩ LOF: {both_ae_lof}")
print(f"Overlap IF ∩ LOF: {both_if_lof}")
print(f"Overlap KM ∩ LOF: {both_km_lof}")
print(f"Overlap AE ∩ IF ∩ KM: {all_three}")
print(f"Overlap AE ∩ IF ∩ KM ∩ LOF: {all_four}")
