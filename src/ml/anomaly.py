# src/ml/anomaly_detection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    """
    Detects anomalies in environmental and occupancy data
    using a hybrid approach: rolling z-score + Isolation Forest.
    """

    def __init__(self, window_size=30, contamination=0.02, random_state=42):
        self.window_size = window_size
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            max_samples="auto",
            n_jobs=-1
        )
        self.features = ["co2", "temp", "humidity", "light", "movement"]

    # ------------------------------------
    def rolling_zscore(self, series):
        """Compute rolling z-score for quick spike detection."""
        rolling_mean = series.rolling(self.window_size, min_periods=1).mean()
        rolling_std = series.rolling(self.window_size, min_periods=1).std()
        zscore = np.abs((series - rolling_mean) / rolling_std)
        return zscore.fillna(0)

    # ------------------------------------
    def preprocess(self, df):
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        # Standardize expected feature names
        possible_map = {
            "co₂": "co2",
            "co2(ppm)": "co2",
            "temperature": "temp"
        }
        df = df.rename(columns=possible_map)

        expected_features = ["temp", "humidity", "co2", "light", "movement"]
        available = [f for f in expected_features if f in df.columns]

        if not available:
            raise ValueError(f"No matching features found in dataset. Columns: {list(df.columns)}")

        # Convert True/False to 1/0
        if "movement" in available:
            df["movement"] = df["movement"].astype(str).map({"True": 1, "False": 0, "1": 1, "0": 0}).fillna(0)

        # Convert numeric
        df[available] = df[available].apply(pd.to_numeric, errors="coerce")

        print(f"✅ Preprocessed {len(df)} rows. Numeric features used: {available}")
        return df


    # ------------------------------------
    def fit(self, df):
        df = self.preprocess(df)
        X = df[self.features].dropna()
        self.model.fit(X)
        # 🔒 Remember exact training column order
        self.feature_names_in_ = list(X.columns)
        print(f"✅ Model trained on {len(X)} samples and {len(self.features)} features.")


    # ------------------------------------
    def detect(self, df):
        """Detect anomalies using both z-score and Isolation Forest."""
        df = self.preprocess(df)

        # Rolling z-score for primary sensors
        for feature in ["co2", "temp", "humidity"]:
            df[f"{feature}_z"] = self.rolling_zscore(df[feature])

        # Combined rolling anomaly flag
        df["z_anomaly"] = ((df["co2_z"] > 3) | (df["temp_z"] > 3) | (df["humidity_z"] > 3)).astype(int)

        # Isolation Forest
        X = df[self.features].fillna(method="ffill")
        df["iforest_anomaly"] = self.model.predict(X)
        df["iforest_anomaly"] = df["iforest_anomaly"].map({1: 0, -1: 1})

        # Final hybrid anomaly score
        df["anomaly"] = ((df["z_anomaly"] + df["iforest_anomaly"]) > 0).astype(int)
        return df

    # ------------------------------------
    def get_anomaly_summary(self, df):
        """Aggregate anomalies by room."""
        summary = (
            df.groupby("room_id")["anomaly"]
            .sum()
            .reset_index()
            .rename(columns={"anomaly": "anomaly_count"})
        )
        summary = summary.sort_values("anomaly_count", ascending=False)
        return summary
    def predict(self, X):
        # ✅ Ensure same feature order for inference
        X = X[self.feature_names_in_]
        return self.model.predict(X)

