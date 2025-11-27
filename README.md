# Blockchain Anomaly Detection using Unsupervised Machine Learning

## 📌 Project Overview

This project analyzes Ethereum token transfer activity and detects *anomalous addresses and transactions* using **unsupervised machine learning models** + a **graph-based structural risk engine**. Because blockchain fraud rarely comes with labeled truth-data, this system instead learns what *normal behavior looks like* — then surfaces deviations that may represent:

* Money laundering
* Drain attacks
* Wash-trading rings
* Bot clusters & MEV behavior
* High-risk liquidity movements

Outputs include anomaly-ranked wallets, suspicious transfer edges, reconstruction distributions, and graph-risk scores.

---

## 🧠 Techniques Used

| Method                                     | Purpose                                                          |
| ------------------------------------------ | ---------------------------------------------------------------- |
| Autoencoder (AE)                           | Learns normal behavior → flags high reconstruction-error wallets |
| Isolation Forest (IF)                      | Detects statistically rare patterns & burst irregularities       |
| KMeans Clustering                          | Finds addresses far from centroid → behavioral outliers          |
| Graph-based Risk Model                     | Highlights hubs, cycles, sinks, extreme connectivity             |

The combination provides **cross-model agreement**, improving anomaly confidence.

---

## 📁 Repository Structure

```
GroupProject/
├── eda_plots/                     # Visuals generated during exploratory analysis
├── EDA_Blockchain.py              # Data exploration, distribution graphs, Lorenz curve
├── blockchain_Unsupervised.py     # AE, IF, + KM scoring implementations
├── GraphUnsupervised.py           # Graph construction + PageRank + structural outliers
├── ae_recon_error_hist_full.png   # Full AE reconstruction error distribution
├── ae_recon_error_hist_zoom.png   # Zoomed tail of high-error anomalies
├── anomaly_landscape.png          # Combined anomaly score visualization
└── .gitignore                     # Standard ignore rules
```

---

## 🚀 Running the Models

### 1. Exploratory Data Analysis (Optional but Recommended)

```bash
python EDA_Blockchain.py
```

Outputs appear in `eda_plots/` and include:

* Token frequency + log-value distributions
* Lorenz inequality curve
* Hourly transfer patterns

### 2. Run Unsupervised Behavioral Models

```bash
python blockchain_Unsupervised.py
```

Generates:

* Autoencoder reconstruction error plots
* Top-ranked behavioral anomalies
* LOF + KM edge-case identifiers

### 3. Graph Structural Risk Detection

```bash
python GraphUnsupervised.py
```

Produces:

* `edge_anomalies.csv`
* `address_risk.csv`
* High-risk hubs, sinks, and transfer chains

---

## 📈 Output Artifacts Included

| File                               | Meaning                                           |
| ---------------------------------- | ------------------------------------------------- |
| `ae_recon_error_hist_full.png`     | Shows normal vs long-tail anomaly separation      |
| `ae_recon_error_hist_zoom.png`     | Clean look at extreme anomalous points            |
| `anomaly_landscape.png`            | Multi-model anomaly landscape visualization       |
| `address_risk.csv` *(generated)*   | Ranked addresses by graph structural deviation    |
| `edge_anomalies.csv` *(generated)* | Suspicious transfers by risk score top-percentile |

---

## 🔮 Future Extensions

* **Expand to longer historical windows** — improves baseline understanding + seasonal detection.
* **Cluster-level anomaly scoring** — identifies coordinated laundering rings & multi-wallet attacks.
* **Real-time streaming version** — produces alerts as transactions hit chain, not post-analysis.
* **Synthetic or labeled validation** — benchmark precision using known attack scenarios.

---

## 🏁 Summary

This repository represents a scalable, label-free fraud detection method for Ethereum transfers — using feature-based anomaly models *plus* a graph-structural risk layer to surface suspicious network behavior.

Perfect for exchanges, compliance analytics, MEV research, and automated on-chain monitoring systems.

---
