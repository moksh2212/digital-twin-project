# src/replay/replay_engine.py

import time
import pandas as pd
from src.representation.building_model import Building

class ReplayEngine:
    def __init__(self, parquet_path, speed=1.0):
        self.dataset = pd.read_parquet(parquet_path)
        self.dataset["date_time"] = pd.to_datetime(self.dataset["date_time"])
        self.dataset = self.dataset.sort_values("date_time")
        

        self.building = Building()
        self.speed = speed  # 1.0 = real time, >1 faster

    def run(self, start_time=None, end_time=None):
        data = self.dataset
        if start_time:
            data = data[data["date_time"] >= pd.to_datetime(start_time)]
        if end_time:
            data = data[data["date_time"] <= pd.to_datetime(end_time)]

        if data.empty:
            print("⚠️ No data found in the specified time range")
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

            # Control replay speed
            if prev_time and self.speed < 1000:
                delta = (row.date_time - prev_time).total_seconds()
                time.sleep(delta / self.speed)
            prev_time = row.date_time

            
            print(f"Processed {i} events – latest state for room {row.room_id}:")
            print(self.building.get_state().get(row.room_id, {}))

        print("Replay finished.")
        return self.building.get_state()
