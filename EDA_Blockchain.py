import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

TOKENS_CSV = "token_labels.csv"
ADDR_COL = "Address"

print("\n[1/9] Loading token labels…")
tokens = pd.read_csv(TOKENS_CSV, dtype=str)
valid_addrs = (
    tokens[ADDR_COL]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .apply(lambda x: x if x.startswith("0x") else "0x" + x)
    .drop_duplicates()
)
valid_arrow = pa.array(valid_addrs.tolist())
print(f"Verified contracts: {len(valid_addrs):,}")


TRANSFERS_PQ = "transfers_20000000.parquet"
cols = [
    "hash", "from", "to", "value", "asset", "category",
    "rawContract.address", "metadata.blockTimestamp", "tokenId",
]

print("[2/9] Reading transfers parquet with filter…")
dataset = ds.dataset(TRANSFERS_PQ, format="parquet")
flt = pc.is_in(ds.field("rawContract.address"), valid_arrow)
table = dataset.to_table(columns=cols, filter=flt)
transfers = table.to_pandas(types_mapper=pd.ArrowDtype)
print("Shape:", transfers.shape)

print("[3/9] Cleaning timestamps & adding time features…")
transfers["ts"] = pd.to_datetime(
    transfers["metadata.blockTimestamp"], errors="coerce", utc=True
)
transfers = transfers.drop(columns=["metadata.blockTimestamp"])  
transfers["date"] = transfers["ts"].dt.date.astype(str)
transfers["hour"] = transfers["ts"].dt.hour.astype("Int8")
transfers["dow"] = transfers["ts"].dt.dayofweek.astype("Int8")
transfers["hour_ts"] = transfers["ts"].dt.floor("h")  

transfers["value"] = pd.to_numeric(transfers["value"], errors="coerce")

print("[4/9] Dataset overview…")
print(transfers.info())
try:
    desc = transfers.describe(include="all", datetime_is_numeric=True)
except TypeError:
    desc = transfers.describe(include="all")
print(desc.T)

print("\nTop tokens by count (20):\n", transfers["asset"].value_counts().head(20))
print("\nTop categories:\n", transfers["category"].value_counts())
print("\nTop senders (10):\n", transfers["from"].value_counts().head(10))
print("\nTop receivers (10):\n", transfers["to"].value_counts().head(10))

print("[5/9] Building plots…")
PLOTS_DIR = Path("eda_plots")
PLOTS_DIR.mkdir(exist_ok=True)

v = transfers["value"].astype("float64")
finite = np.isfinite(v)
v_ws = v[finite].clip(upper=np.nanquantile(v[finite], 0.999))
plt.figure(figsize=(8,5))
plt.hist(np.log1p(v_ws), bins=100, alpha=0.8)
plt.title("Distribution of log(1+value)")
plt.xlabel("log(1 + value)")
plt.ylabel("Frequency")
plt.savefig(PLOTS_DIR / "value_dist_log.png")
plt.close()

by_day = transfers.groupby("date").size()
plt.figure(figsize=(10,4))
plt.plot(pd.to_datetime(by_day.index), by_day.values)
plt.title("Transfers per Day")
plt.xlabel("Date")
plt.ylabel("Count")
plt.savefig(PLOTS_DIR / "transfers_per_day.png")
plt.close()

by_hour = transfers.groupby("hour").size()
plt.figure(figsize=(8,4))
sns.barplot(x=by_hour.index, y=by_hour.values)
plt.title("Transfers by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.savefig(PLOTS_DIR / "transfers_by_hour.png")
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(
    data=transfers.dropna(subset=["value"]).assign(log1p_value=np.log1p(transfers["value"])),
    x="category", y="log1p_value"
)
plt.title("log(1+value) by category")
plt.savefig(PLOTS_DIR / "value_by_category.png")
plt.close()

counts = transfers["category"].value_counts(dropna=False)
plt.figure(figsize=(6,6))
plt.pie(
    counts.values,
    labels=None,              
    autopct=None,             
    startangle=90
)
plt.title("Token Type Share (by transfers)")
plt.legend(
    labels=[f"{cat} ({val/len(transfers)*100:.2f}%)" for cat, val in counts.items()],
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)
plt.savefig(PLOTS_DIR / "token_type_share.png", bbox_inches="tight")


TOP_N = 10
top_tokens = transfers["asset"].value_counts().head(TOP_N).index
hourly_token = (
    transfers[transfers["asset"].isin(top_tokens)]
    .groupby(["hour_ts", "asset"])  
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
plt.figure(figsize=(12,6))
sns.heatmap(np.log1p(hourly_token), cmap="viridis")
plt.title(f"Hourly Token Activity Heatmap (Top {TOP_N}) – log(1+count)")
plt.xlabel("Token")
plt.ylabel("Hour")
plt.savefig(PLOTS_DIR / "hourly_token_heatmap.png")
plt.close()

plt.figure(figsize=(12,5))
for tok in top_tokens:
    if tok in hourly_token.columns:
        plt.plot(hourly_token.index, hourly_token[tok], label=tok)
plt.legend(ncol=2)
plt.title(f"Hourly Transfers – Top {TOP_N} tokens")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.savefig(PLOTS_DIR / "top_tokens_hourly.png")
plt.close()

finite_rows = transfers[finite]
SAMPLE = min(200_000, len(finite_rows))
samp = finite_rows.sample(SAMPLE, random_state=42) if SAMPLE < len(finite_rows) else finite_rows
plt.figure(figsize=(12,5))
plt.scatter(samp["ts"], np.log1p(samp["value"].astype("float64")), s=2, alpha=0.2)
plt.title("Value vs. Time (log scale)")
plt.xlabel("Timestamp")
plt.ylabel("log(1+value)")
plt.savefig(PLOTS_DIR / "value_vs_time_scatter.png")
plt.close()

from_counts = transfers["from"].value_counts()
to_counts = transfers["to"].value_counts()
addr_activity = from_counts.add(to_counts, fill_value=0).astype("float64").sort_values()
activity = addr_activity.values
if activity.size > 0 and activity.sum() > 0:
    cum_activity = np.cumsum(activity)
    lorenz_y = np.concatenate([[0.0], cum_activity / cum_activity[-1]])
    lorenz_x = np.linspace(0.0, 1.0, num=lorenz_y.size)
    gini = 1.0 - 2.0 * np.trapz(lorenz_y, lorenz_x)
    shares = activity / activity.sum()
    hhi = np.sum(shares ** 2)

    plt.figure(figsize=(6,6))
    plt.plot(lorenz_x, lorenz_y, label="Lorenz curve")
    plt.plot([0,1], [0,1], linestyle="--", color="gray", label="Equality line")
    plt.title(f"Address Activity Lorenz Curve\nGini={gini:.3f}, HHI={hhi:.4f}")
    plt.xlabel("Cumulative share of addresses")
    plt.ylabel("Cumulative share of transfers")
    plt.legend(loc="lower right")
    plt.savefig(PLOTS_DIR / "lorenz_address_activity.png")
    plt.close()

    print(f"Address concentration – Gini: {gini:.3f}, HHI: {hhi:.4f}")
else:
    print("Lorenz skipped: no address activity counts available.")

print("[6/9] Saving compact cleaned parquet…")
out_cols = [
    "hash","from","to","value","asset","category","rawContract.address",
    "ts","date","hour","hour_ts","dow","tokenId"
]
transfers[out_cols].to_parquet("transfers_verified_small.parquet", index=False)


print("\n[7/9] Slide‑ready stats:")
print("Rows:", f"{len(transfers):,}")
print("Unique hashes:", f"{transfers['hash'].nunique():,}")
print("Unique senders:", f"{transfers['from'].nunique():,}")
print("Unique receivers:", f"{transfers['to'].nunique():,}")
print("Unique tokens (asset):", f"{transfers['asset'].nunique():,}")
print("Unique contracts:", f"{transfers['rawContract.address'].nunique():,}")

print("\nValue stats (finite, 99.9% winsorized):")
print(pd.Series(v_ws).describe(percentiles=[0.5,0.9,0.99]))

print("\n[8/9] New plots saved to ./eda_plots:")
print(" - hourly_token_heatmap.png")
print(" - top_tokens_hourly.png")
print(" - value_vs_time_scatter.png")
print(" - token_type_share.png")
print(" - lorenz_address_activity.png")
print("[9/9] EDA complete – artifacts saved and parquet written.")
