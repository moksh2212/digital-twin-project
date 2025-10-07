# src/ml/room_clustering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RoomClusteringModel:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.features = ["temp", "humidity", "co2", "light", "movement"]
        self.trained = False

    def preprocess(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        # Handle variations
        rename_map = {
            "temperature": "temp",
            "temp": "temp",
            "humid": "humidity",
            "co2": "co2",
            "light": "light",
            "movement": "movement"
        }
        df = df.rename(columns=rename_map)

        # Coerce to numeric safely
        for f in self.features:
            if f in df.columns:
                df[f] = pd.to_numeric(df[f], errors="coerce")

        # Convert movement to 0/1
        if "movement" in df.columns:
            df["movement"] = df["movement"].astype(str).str.lower().map({
                "true": 1, "false": 0, "1": 1, "0": 0
            }).fillna(0)

        # Drop only if *all* features missing (less aggressive)
        df = df.dropna(subset=self.features, how="all")

        # Just warn if empty
        if df.empty:
            print("⚠️ Warning: no valid rows after preprocessing — check feature names/values.")

        return df


    def fit(self, df):
            df = self.preprocess(df)
            X = df[self.features]
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled)
            self.trained = True
            print(f"✅ Trained K-Means model with {self.n_clusters} clusters on {len(X)} readings.")
            return df

    def predict(self, df):
            if not self.trained:
                raise RuntimeError("Model not trained. Call fit() first.")
            df = self.preprocess(df)
            X_scaled = self.scaler.transform(df[self.features])
            return self.model.predict(X_scaled)

