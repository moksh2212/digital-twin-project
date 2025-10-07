# src/representation/building_model.py

class Room:
    def __init__(self, room_id):
        self.room_id = room_id
        self.state = {}   # {sensor_type: (value, timestamp)}

    def update_state(self, sensor_type, value, timestamp):
        self.state[sensor_type] = {"value": value, "timestamp": timestamp}

    def __repr__(self):
        return f"<Room {self.room_id}, sensors={list(self.state.keys())}>"

# src/representation/building_model.py
class Building:
    def __init__(self):
        self.state = {}  # { room_id: {sensor_values...} }

    def process_event(self, event):
        room = event["room_id"]
        values = event["value"]
        if room not in self.state:
            self.state[room] = {}
        self.state[room].update(values)

    def update_ml_results(self, room_id, anomaly=None, health=None, cluster=None):
        if room_id not in self.state:
            self.state[room_id] = {}
        if anomaly is not None:
            self.state[room_id]["anomaly_flag"] = anomaly
        if health is not None:
            self.state[room_id]["health_status"] = health
        if cluster is not None:
            self.state[room_id]["cluster_label"] = cluster

    def get_state(self):
        return self.state

