import os
import numpy as np
import pandas as pd
import glob
import re

import tempfile

import tqdm

class DataLoader:
    def __init__(self, rep_dir, args):
        self.output_dir = os.path.join(os.getcwd(), "data", "input", "dst")

        self.data = self.load_data(rep_dir, args)
        
        filename = os.path.basename(rep_dir)
        args.output = os.path.join(self.output_dir, filename)
        os.makedirs(args.output, exist_ok=True)
        
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        
        self.data["filename"] = filename
        self.data["temp_dir"] = tempfile.mkdtemp()
        
        self.save_meta()
        self.save_terrain()
        
    def save_meta(self):
        with open(os.path.join(self.output_dir, self.data["filename"], "meta.txt"), "w") as f:
            f.write(f"map_name: {self.data['map_name']}\n")
            f.write(f"width: {self.data['width']}\n")
            f.write(f"height: {self.data['height']}\n")
            f.write(f"game_length: {self.data['game_length']}\n")
            f.write(f"resolution_frame: {self.data['resolution_frame']}\n")
            for k, v in self.data["players_data"].items():
                if k == "neutral": continue
                f.write(f"{k}: {v['name']}\n")
    
    def save_terrain(self):
        np.save(os.path.join(self.data["temp_dir"], "terrain.npy"), self.data["terrain"])
        
    def _open_file_with_fallback(self, file_path):
        """
        Tries to open a file without specifying encoding. 
        If it fails, retries with cp949 encoding.
        """
        try:
            with open(file_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="cp949") as f:
                return [line.strip() for line in f.readlines()]

    def load_meta(self, path):
        print("Load meta...", end="")

        meta = self._open_file_with_fallback(os.path.join(path, "meta"))
        meta = dict(zip(meta[0].split(","), meta[1].split(",")))

        meta["width"] = int(meta["width"])
        meta["height"] = int(meta["height"])
        meta["game_length"] = int(meta["game_length"])

        if "players_data" in meta:
            pattern = re.compile(r"\[((?:[^\[\];]+|\[[^\]]*\])+);([^;]+);([^\]]+)\]")
            players_list = []

            for match in pattern.finditer(meta["players_data"]):
                name, race, start_loc = match.groups()
                players_list.append({
                    "name": name.strip(),
                    "race": race.strip(),
                    "start_location": start_loc.strip()
                })

            meta["players_data"] = {f"player_{i+1}": p for i, p in enumerate(players_list)}
            meta["players_data"]["neutral"] = {
                "name": "Neutral", "race": None, "start_location": None
            }

        print("Done")
        return meta

    def load_terrain(self, path, meta):
        print("Load terrain...", end="")
        
        terrain = self._open_file_with_fallback(os.path.join(path, "terrain"))
        terrain = np.array([np.asarray(line.split(","), dtype=int) for line in terrain])
    
        print("Done")
        return terrain

    def load_vision(self, path, meta):
        print("Load vision...", end="")
        
        vision = self._open_file_with_fallback(os.path.join(path, "vision"))
        
        print("Done")
        return vision

    def load_event(self, path, meta):
        print("Load event...", end="")
        events = self._open_file_with_fallback(os.path.join(path, "event"))
        
        iterable = tqdm.tqdm(
            events[1:], 
            desc="Load event...", 
            miniters=max(1, len(events[1:]) // 20)
        )

        events = [
            {
                "frame": frame, 
                "ID": id, 
                "x": x, 
                "y": y, 
                "player": player, 
                "unit_type": unit_type, 
                "event_type": event_type,
                "target_id": target_id,
                "target_x": target_x,
                "target_y": target_y,
            }
            for event in iterable
            for frame, id, x, y, player, unit_type, event_type, target_id, target_x, target_y in [event.split(",")]
            for frame, x, y in [map(int, [frame, x, y])]
        ]

        return events

    def compute_resolution_frame(self, df: pd.DataFrame, game_length: int) -> int:
        if df.empty:
            return game_length + 1

        players = [p for p in df['player'].unique() if p != 'Neutral']
        if len(players) < 2:
            return game_length + 1

        last_seen = {}
        for p in players:
            sub = df.loc[df['player'] == p, 'frame']
            if not sub.empty:
                last_seen[p] = int(sub.max())

        if len(last_seen) < 2:
            return game_length + 1

        cut = min(last_seen.values()) + 1  # last used frame + 1
        return min(cut, game_length + 1)

    def load_state(self, path):
        print("Load state...", end="")
        dataframe = pd.read_csv(os.path.join(path, "state"),
                                skiprows=1,
                                names=[
                                'frame', 'player', 'race', 'player_color', 'name', 'ID', 'x', 'y',
                                'top', 'bottom', 'left', 'right', 'HP', 'max_HP', 'shield', 'max_shield',
                                'energy', 'max_energy'
                                ],
                                dtype={
                                    'frame': int, 'x': int, 'y': int,
                                        'top': int, 'bottom': int, 'left': int, 'right': int,
                                        'HP': int, 'max_HP': int, 'shield': int, 'max_shield': int,
                                        'energy': int, 'max_energy': int,
                                        'player': str, 'race': str, 'player_color': str, 'name': str, 'ID': str
                                        }
                                )
        print("Done")
        return dataframe

    def load_data(self, file_path, args):
        meta = self.load_meta(file_path)
        state_dataframe = self.load_state(file_path)
        
        total_meta = int(meta["game_length"]) + 1
        resolution_frame = self.compute_resolution_frame(
            state_dataframe, game_length=int(meta["game_length"])
        )
        
        return {
            "map_name": meta["map_name"],
            "width": meta["width"],
            "height": meta["height"],
            "game_length": meta["game_length"],
            "players_data": meta["players_data"],
            "terrain": self.load_terrain(file_path, meta),
            "vision_raw": self.load_vision(file_path, meta),
            "state_raw": state_dataframe,
            "resolution_frame": min(resolution_frame, total_meta),
        }