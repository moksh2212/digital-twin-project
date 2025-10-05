# src/representation/building_model.py

class Room:
    def __init__(self, room_id):
        self.room_id = room_id
        self.state = {}   # {sensor_type: (value, timestamp)}

    def update_state(self, sensor_type, value, timestamp):
        self.state[sensor_type] = {"value": value, "timestamp": timestamp}

    def __repr__(self):
        return f"<Room {self.room_id}, sensors={list(self.state.keys())}>"

class Building:
    def __init__(self):
        self.rooms = {}

    def process_event(self, event):
        """event = dict with keys {room_id, sensor, value, timestamp}"""
        room = self.rooms.setdefault(event["room_id"], Room(event["room_id"]))
        room.update_state(event["sensor"], event["value"], event["timestamp"])

    def get_state(self):
        """Return current snapshot of building state"""
        snapshot = {}
        for room_id, room in self.rooms.items():
            snapshot[room_id] = {s: v["value"] for s, v in room.state.items()}
        return snapshot
