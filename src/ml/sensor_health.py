# src/ml/sensor_health.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class SensorHealthModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=8
        )
        self.scaler = StandardScaler()
        self.features = ["voltage", "rssi", "snr"]
        self.trained = False

    def preprocess(self, df):
        df = df.copy()
        # Convert to numeric safely
        df[self.features] = df[self.features].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=self.features)

        # Define a pseudo "health" label
        # Example: if voltage < 3.5V OR rssi < -100 OR snr < 5 → unhealthy
        df["health_status"] = np.where(
            (df["voltage"] < 3.5) | (df["rssi"] < -100) | (df["snr"] < 5),
            0,  # unhealthy
            1   # healthy
        )
        return df

    def fit(self, df):
        df = self.preprocess(df)
        X = df[self.features]
        y = df["health_status"]

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.trained = True

        preds = self.model.predict(X_test)
        print("✅ Sensor Health Model trained successfully.\n")
        print(classification_report(y_test, preds))
        return df

    def predict(self, df):
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        df = self.preprocess(df)
        X_scaled = self.scaler.transform(df[self.features])
        df["predicted_health"] = self.model.predict(X_scaled)
        return df
