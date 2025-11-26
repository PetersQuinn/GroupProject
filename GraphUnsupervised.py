import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PATH = "transfers_verified_small.parquet"
TOP_K = 30             # how many suspicious transfers to print
SUBSAMPLE = None       # e.g., 300000 to speed up if needed, else None (use all)

# Whitelist: well-known routers/aggregators/exchanges/infra (public mainnet addresses)
#reduce false positives from structurally-weird-but-benign hubs
WHITELIST = {
    # 1inch routers / aggregation
    "0x111111125421ca6dc452d289314280a0f8842a65",  # 1inch: Aggregation Router v4
    "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch: Aggregation Router v5
    # Uniswap routers
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 Router
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 Router
    "0x68b3465833fb72a70ecdF485E0e4C7bD8665Fc45",  # Uniswap V3 Router 2
    # 0x (Matcha) Exchange Proxy
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
    # Paraswap Augustus
    "0xdef171fe48cf0115b1d80b88dc8eab59176fee57",
    # CowSwap settlement
    "0x9008d19f58aabd9ed0d60971565aa8510560ab41",
    # Known splitter/relayer seen in many datasets (e.g., 1inch helper)
    "0x74de5d4fcbf63e00296fd95d33236b9794016631",
}

# Sharding controls
USE_TIME_SHARD = True  # set False to shard only by token_contract
WINDOW_HOURS = 6       # time window size when sharding by time (hyperparameter)

# ---------------------------
# 1) Load & basic clean
# ---------------------------

df = pd.read_parquet(PATH)

# standardize column names we'll use

df = df.rename(columns={
    "from": "src",
    "to": "dst",
    "ts": "timestamp",
    "rawContract.address": "token_contract"
})
keep = ["hash","src","dst","value","asset","category","token_contract","timestamp","hour","dow"]
df = df[keep].copy()

# minimal cleaning

df = df.dropna(subset=["src","dst","value","timestamp"]) 
df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
df["hour"]  = pd.to_numeric(df["hour"], errors="coerce").fillna(-1).astype(int)
df["dow"]   = pd.to_numeric(df["dow"],  errors="coerce").fillna(-1).astype(int)

# normalize nullable strings

df["asset"] = df["asset"].astype("string").fillna("UNK")
df["category"] = df["category"].astype("string").fillna("")
df["token_contract"] = df["token_contract"].astype("string").fillna("UNK")

if SUBSAMPLE:
    df = df.head(SUBSAMPLE)

# ---------------------------
# 2) Build directed address graph (aggregate parallel edges)
# ---------------------------

edge_agg = df.groupby(["src","dst"], as_index=False)["value"].sum().rename(columns={"value":"w"})
G = nx.DiGraph()
G.add_weighted_edges_from(edge_agg[["src","dst","w"]].itertuples(index=False, name=None))

# ---------------------------
# 3) Node-level graph stats
# ---------------------------

deg_in   = dict(G.in_degree())
deg_out  = dict(G.out_degree())
str_in   = dict(G.in_degree(weight="weight"))
str_out  = dict(G.out_degree(weight="weight"))

try:
    pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=100)
except Exception:
    pr = {n: 0.0 for n in G.nodes()}

# Optional: simple clustering on undirected projection (cheap proxy)
UG = G.to_undirected()
try:
    cc = nx.clustering(UG, weight="weight")
except Exception:
    cc = {n: 0.0 for n in G.nodes()}

# ---------------------------
# 4) Edge (transfer) feature table
# ---------------------------

X = df.copy()


def m(series, dct, default=0.0):
    return series.map(dct).fillna(default)

# endpoint stats
X["src_deg_in"]  = m(X["src"], deg_in)
X["src_deg_out"] = m(X["src"], deg_out)
X["dst_deg_in"]  = m(X["dst"], deg_in)
X["dst_deg_out"] = m(X["dst"], deg_out)

X["src_str_in"]  = m(X["src"], str_in)
X["src_str_out"] = m(X["src"], str_out)
X["dst_str_in"]  = m(X["dst"], str_in)
X["dst_str_out"] = m(X["dst"], str_out)

X["src_pr"]      = m(X["src"], pr)
X["dst_pr"]      = m(X["dst"], pr)
X["src_cc"]      = m(X["src"], cc)
X["dst_cc"]      = m(X["dst"], cc)

# value transforms
X["value_log1p"] = np.log1p(X["value"])

# robust categorical flags (handle pyarrow strings + NA)
cat_norm   = X["category"].astype("string").str.lower().fillna("")
asset_norm = X["asset"].astype("string").str.upper().fillna("")

X["is_erc20"]  = (cat_norm == "erc20").astype("int8")
X["is_erc721"] = (cat_norm == "erc721").astype("int8")
X["is_eth"]    = (asset_norm == "ETH").astype("int8")

# per-(asset,token_contract) normalization to capture unusual amount
grp = X.groupby(["asset","token_contract"])["value"]
X["value_mu"] = grp.transform("mean")
X["value_sd"] = grp.transform("std").replace(0, np.nan)
X["value_z"]  = (X["value"] - X["value_mu"]) / X["value_sd"]
X["value_z"]  = X["value_z"].fillna(0.0)

# feature set for anomaly scoring
feature_cols = [
    "src_deg_in","src_deg_out","dst_deg_in","dst_deg_out",
    "src_str_in","src_str_out","dst_str_in","dst_str_out",
    "src_pr","dst_pr","src_cc","dst_cc",
    "value","value_log1p","value_z",
    "is_erc20","is_erc721","is_eth",
    "hour","dow"
]

# ---------------------------
# 5) Sharded unsupervised anomaly scoring (IsolationForest)
# ---------------------------

# Whitelist filtering: drop edges where either endpoint is infra
mask_infra = X["src"].isin(WHITELIST) | X["dst"].isin(WHITELIST)
X = X.loc[~mask_infra].copy()

if USE_TIME_SHARD:
    X["_window"] = pd.to_datetime(X["timestamp"]).dt.floor(f"{WINDOW_HOURS}H")
    shard_cols = ["token_contract", "_window"]
else:
    shard_cols = ["token_contract"]

X["anomaly_score"] = np.nan

for keys, idx in X.groupby(shard_cols, sort=False).groups.items():
    sl = X.loc[idx, feature_cols].astype(float).fillna(0.0)
    if len(sl) < 200:  # skip tiny shards to avoid unstable fits
        X.loc[idx, "anomaly_score"] = 0.0
        continue
    scaler = StandardScaler()
    Z = scaler.fit_transform(sl.values)
    iso = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Z)
    X.loc[idx, "anomaly_score"] = -iso.score_samples(Z)  # higher = riskier

X["anomaly_score"] = X["anomaly_score"].fillna(0.0)

# ---------------------------
#Print top suspicious transfers & save CSVs
# ---------------------------

top_edges = X.sort_values("anomaly_score", ascending=False).head(TOP_K)
cols_show = [
    "hash","timestamp","src","dst","asset","category","token_contract",
    "value","value_z","src_deg_out","dst_deg_in","src_pr","dst_pr","anomaly_score"
]
print("\nTop suspicious transfers (unsupervised, sharded):")
print(top_edges[cols_show].to_string(index=False, max_colwidth=80))

# edge-level output
edge_out_cols = [
    "hash","timestamp","src","dst","asset","category","token_contract",
    "value","anomaly_score"
] + feature_cols
X.sort_values("anomaly_score", ascending=False).to_csv("edge_anomalies.csv", index=False, columns=edge_out_cols)
print("\nSaved: edge_anomalies.csv")

src_roll = X.groupby("src").agg(
    n_src_edges=("src","size"),
    src_risk_mean=("anomaly_score","mean"),
    src_risk_max=("anomaly_score","max"),
    src_deg_out=("src_deg_out","max"),
    src_pr=("src_pr","max")
).rename_axis("address").reset_index()

dst_roll = X.groupby("dst").agg(
    n_dst_edges=("dst","size"),
    dst_risk_mean=("anomaly_score","mean"),
    dst_risk_max=("anomaly_score","max"),
    dst_deg_in=("dst_deg_in","max"),
    dst_pr=("dst_pr","max")
).rename_axis("address").reset_index()

addr = pd.merge(src_roll, dst_roll, on="address", how="outer").fillna(0)
addr["combined_risk"] = addr[["src_risk_max","dst_risk_max","src_risk_mean","dst_risk_mean"]].max(axis=1)

addr.sort_values("combined_risk", ascending=False).to_csv("address_risk.csv", index=False)
print("Saved: address_risk.csv")

print("\nTop suspicious addresses (by combined_risk):")
print(addr.sort_values("combined_risk", ascending=False).head(20).to_string(index=False, max_colwidth=80))


from umap import UMAP
import matplotlib.pyplot as plt


feat_matrix = X[feature_cols].astype(float).fillna(0.0).values
scores = X["anomaly_score"].values

# Optional: downsample for faster plotting if you have millions of rows
MAX_POINTS = 50000
if len(feat_matrix) > MAX_POINTS:
    # sample without replacement but preserve highest-risk points
    # 1) take all of the top high-score points
    top_n = min(5000, len(feat_matrix))
    top_idx = np.argsort(scores)[-top_n:]
    # 2) sample the rest
    remaining_idx = np.setdiff1d(np.arange(len(feat_matrix)), top_idx)
    rng = np.random.default_rng(42)
    sample_rest = rng.choice(remaining_idx, size=MAX_POINTS - top_n, replace=False)

    keep_idx = np.concatenate([top_idx, sample_rest])
    feat_matrix = feat_matrix[keep_idx]
    scores = scores[keep_idx]

# Embed to 2D using UMAP
umap = UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
)
embedding = umap.fit_transform(feat_matrix)

# Plot: each point is a transfer, colored by anomaly score
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    s=5,
    c=scores,
)
cbar = plt.colorbar(scatter)
cbar.set_label("Anomaly score (IsolationForest)", rotation=270, labelpad=15)

plt.title("Ethereum transfers in graph-augmented feature space\ncolored by anomaly score")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("anomaly_landscape.png", dpi=300)
plt.show()

print('Saved 2D anomaly landscape figure as "anomaly_landscape.png"')
