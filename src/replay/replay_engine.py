# src/replay/replay_engine.py
import time
import pandas as pd
import numpy as np
from src.representation.building_model import Building
from src.ml.anomaly import AnomalyDetector
from src.ml.sensor_health import SensorHealthModel
from src.ml.room_clustering import RoomClusteringModel


class ReplayEngine:
    def __init__(self, parquet_path, speed=1.0):
        self.dataset = pd.read_parquet(parquet_path).sort_values("date_time")
        self.building = Building()
        self.speed = speed

        # Initialize ML models
        self.anomaly_model = AnomalyDetector()
        self.health_model = SensorHealthModel()
        self.cluster_model = RoomClusteringModel()

        # Fit models once before replay
        print("Training ML models...")
        self.anomaly_model.fit(self.dataset)
        self.health_model.fit(self.dataset)
        self.cluster_model.fit(self.dataset)
        print("✅ All ML models trained.\n")

    # ------------------------------------------------------------
    # Helper: safely extract scalar from any prediction output
    # ------------------------------------------------------------
    def _to_scalar(self, pred):
        """Convert any prediction output type (np, list, Series, DataFrame, scalar) → single value"""
        if isinstance(pred, (list, np.ndarray)):
            return float(pred[0])
        elif isinstance(pred, pd.Series):
            return float(pred.iloc[0])
        elif isinstance(pred, pd.DataFrame):
            return float(pred.iloc[0, 0])
        else:
            return float(pred)

    # ------------------------------------------------------------
    # Main replay loop
    # ------------------------------------------------------------
    def run(self, start_time=None, end_time=None, log_every=5000):
        data = self.dataset.copy()

        # Ensure datetime format
        data["date_time"] = pd.to_datetime(data["date_time"], errors="coerce")

        if start_time:
            data = data[data["date_time"] >= pd.to_datetime(start_time)]
        if end_time:
            data = data[data["date_time"] <= pd.to_datetime(end_time)]

        if data.empty:
            print("⚠️ No data found in the specified range.")
            return

        prev_time = None
        for i, row in enumerate(data.itertuples(index=False), start=1):
            event = {
                "timestamp": row.date_time,
                "room_id": row.room_id,
                "sensor": "sensor_event",
                "value": row._asdict()
            }
            self.building.process_event(event)

            # Construct features in consistent order
            features = pd.DataFrame([{
                "room_id": getattr(row, "room_id", None),
                "temp": getattr(row, "temp", np.nan),
                "humidity": getattr(row, "humidity", np.nan),
                "co2": getattr(row, "co2", np.nan),
                "light": getattr(row, "light", np.nan),
                "movement": int(bool(getattr(row, "movement", 0))),
                "voltage": getattr(row, "voltage", np.nan),
                "rssi": getattr(row, "rssi", np.nan),
                "snr": getattr(row, "snr", np.nan)
            }])
            # Force all relevant numeric features to numeric dtype
            for col in ["temp", "humidity", "co2", "light", "movement", "voltage", "rssi", "snr"]:
                if col in features.columns:
                    features[col] = pd.to_numeric(features[col], errors="coerce")

            # Drop non-numeric or missing values
            numeric_features = features[
                ["temp", "humidity", "co2", "light", "movement", "voltage", "rssi", "snr"]
            ].apply(pd.to_numeric, errors="coerce").dropna(how="any")

            if numeric_features.empty:
                continue

            # Run live predictions safely
            anomaly_flag = int(self._to_scalar(self.anomaly_model.predict(numeric_features)))
            health_df = self.health_model.predict(numeric_features)
            health_pred = int(health_df["predicted_health"].iloc[0])

            # For clustering, we need room_id
            cluster_label = int(self._to_scalar(self.cluster_model.predict(features)))


            # Update building digital twin state
            self.building.update_ml_results(
                row.room_id,
                anomaly=anomaly_flag,
                health=health_pred,
                cluster=cluster_label
            )

            # Optional delay for realistic replay
            if prev_time and self.speed < 1000:
                delta = (row.date_time - prev_time).total_seconds()
                time.sleep(delta / self.speed)
            prev_time = row.date_time

            # Periodic log
            if i % log_every == 0:
                print(f"Processed {i} events | Latest room {row.room_id}:")
                print(self.building.get_state().get(row.room_id, {}))

        print("🎬 Replay complete.")
        return self.building.get_state()
