# -*- coding: utf-8 -*-
"""
The Evolution of Music: 1921-2020
Dataset: Spotify Dataset 1921-2020, 160k+ Tracks (yamaerenay on Kaggle)

Sections
  1. Load the data
  2. Clean the data
  3. Trends over time
  4. Predict a song's decade (Random Forest vs Logistic Regression)
  5. Robustness check: remove spoken-word tracks
  6. Cluster songs by sound (K-Means, K = 3 and K = 6)

Every chart is saved to the "outputs" folder BEFORE it is shown.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, silhouette_score)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def save_and_show(filename):
    """Save the current figure first, then show it (the order matters)."""
    plt.savefig(os.path.join(OUT, filename), dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------
path = kagglehub.dataset_download("yamaerenay/spotify-dataset-1921-2020-160k-tracks")
print(os.listdir(path))

df = pd.read_csv(os.path.join(path, "data.csv"))
print(df.shape)
print(df.isnull().sum().sum(), "missing values")
print(df.duplicated().sum(), "duplicate rows")
print("Years:", df["year"].min(), "to", df["year"].max())

# ---------------------------------------------------------------
# 2. CLEAN THE DATA
# ---------------------------------------------------------------
df = df.drop_duplicates()
data = df.copy()

data["decade"] = (data["year"] // 10) * 10          # 1967 -> 1960
data["duration_min"] = data["duration_ms"] / 60000  # milliseconds -> minutes

print((data["tempo"] == 0).sum(), "songs with tempo = 0")
print((data["duration_min"] > 15).sum(), "songs longer than 15 minutes")
print((data["duration_min"] < 0.5).sum(), "songs shorter than 30 seconds")

data = data[data["tempo"] > 0]
data = data[(data["duration_min"] >= 0.5) & (data["duration_min"] <= 15)]
print(data.shape)
print(data.groupby("decade").size())

# 2020 is a single year, so it is left out of decade-level analysis
data_decades = data[data["decade"] < 2020].copy()

# ---------------------------------------------------------------
# 3. TRENDS OVER TIME
# ---------------------------------------------------------------
trend_features = ["danceability", "energy", "valence", "acousticness",
                  "loudness", "duration_min", "tempo"]

yearly = data.groupby("year")[trend_features].mean()

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes = axes.flatten()
for ax, f in zip(axes, trend_features):
    ax.plot(yearly.index, yearly[f])
    ax.set_title(f)
    ax.set_xlabel("Year")
axes[-1].axis("off")
plt.tight_layout()
save_and_show("trends_by_year.png")

decade_means = data_decades.groupby("decade")[trend_features].mean().round(2)
print(decade_means)
decade_means.to_csv(os.path.join(OUT, "decade_averages.csv"))

plt.figure(figsize=(8, 6))
sns.heatmap(data[trend_features].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation between features")
save_and_show("feature_correlation.png")

# ---------------------------------------------------------------
# 4. PREDICT A SONG'S DECADE
# ---------------------------------------------------------------
model_features = ["danceability", "energy", "valence", "acousticness",
                  "instrumentalness", "liveness", "speechiness", "loudness",
                  "tempo", "duration_min", "key", "mode", "explicit"]

X = data_decades[model_features]
y = data_decades["decade"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)
logreg_acc = accuracy_score(y_test, logreg.predict(X_test))
print("Logistic Regression accuracy:", logreg_acc)

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

error = np.abs(rf_pred - y_test.values)
rf_exact = (error == 0).mean()
rf_within10 = (error <= 10).mean()
print("Exactly right:", rf_exact)
print("Within one decade:", rf_within10)

ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, cmap="Blues")
plt.title("Random Forest: actual vs predicted decade")
save_and_show("confusion_matrix.png")

importance = pd.Series(rf.feature_importances_, index=model_features).sort_values()
importance.plot(kind="barh", figsize=(8, 5), title="Feature importance")
save_and_show("feature_importance.png")

# ---------------------------------------------------------------
# 5. ROBUSTNESS CHECK: REMOVE SPOKEN-WORD TRACKS
# ---------------------------------------------------------------
music_only = data_decades[data_decades["speechiness"] <= 0.66]
print(len(data_decades) - len(music_only), "tracks removed")

X2 = music_only[model_features]
y2 = music_only["decade"]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, stratify=y2, random_state=42)

rf2 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf2.fit(X2_train, y2_train)
pred2 = rf2.predict(X2_test)

err2 = np.abs(pred2 - y2_test.values)
music_exact = accuracy_score(y2_test, pred2)
music_within10 = (err2 <= 10).mean()
print("Music only - exact:", music_exact)
print("Music only - within one decade:", music_within10)
print(classification_report(y2_test, pred2))

results = pd.DataFrame({
    "model": ["Logistic Regression", "Random Forest", "Random Forest (music only)"],
    "exact_accuracy": [logreg_acc, rf_exact, music_exact],
    "within_one_decade": [np.nan, rf_within10, music_within10],
})
results.round(4).to_csv(os.path.join(OUT, "model_results.csv"), index=False)

# ---------------------------------------------------------------
# 6. CLUSTER SONGS BY SOUND
# ---------------------------------------------------------------
cluster_features = ["danceability", "energy", "valence", "acousticness",
                    "instrumentalness", "liveness", "speechiness",
                    "loudness", "tempo", "duration_min"]

cdata = data_decades.copy()
X_scaled = StandardScaler().fit_transform(cdata[cluster_features])

# 6a. Choose K using a 20,000-song sample (the silhouette score is slow)
rng = np.random.RandomState(42)
X_sample = X_scaled[rng.choice(len(X_scaled), 20000, replace=False)]

ks = range(2, 11)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_sample)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_sample, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertias, marker="o"); ax[0].set_title("Elbow (inertia)")
ax[1].plot(ks, sils, marker="o");     ax[1].set_title("Silhouette score")
for a in ax:
    a.set_xlabel("Number of clusters (k)")
save_and_show("choose_k.png")

# 6b. Final clusterings. Each K gets its OWN column and its OWN files,
#     so one run can never overwrite the other.
for k in (3, 6):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    col = f"cluster_k{k}"
    cdata[col] = km.fit_predict(X_scaled)

    profile = cdata.groupby(col)[cluster_features].mean().round(2)
    print(f"\nK = {k}")
    print(profile)
    print(cdata[col].value_counts().sort_index())
    profile.to_csv(os.path.join(OUT, f"cluster_profiles_k{k}.csv"))

    share = pd.crosstab(cdata["decade"], cdata[col], normalize="index")
    share.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab10")
    plt.title(f"Share of each cluster by decade (K = {k})")
    plt.ylabel("Share of songs")
    plt.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_and_show(f"cluster_share_by_decade_k{k}.png")

# 6c. What is in the spoken-word cluster? (K = 6)
profile6 = cdata.groupby("cluster_k6")[cluster_features].mean()
spoken_cluster = profile6["speechiness"].idxmax()
print("\nSpoken-word cluster is number", spoken_cluster)
print(cdata[cdata["cluster_k6"] == spoken_cluster]
      [["name", "artists", "year", "speechiness"]]
      .sample(20, random_state=1))
