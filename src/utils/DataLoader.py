import os
import numpy as np
import pandas as pd
import glob
import re

import tempfile

import tqdm

class DataLoader:
    def __init__(self, rep_file, args):
        self.output_dir = args.output if args.output else os.path.join(os.getcwd(), 'results')

        self.data = self.load_data(rep_file, args)
        
        filename = os.path.basename(rep_file)
        args.output = os.path.join(self.output_dir, filename)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        
        self.data['filename'] = filename
        self.data['temp_dir'] = tempfile.mkdtemp()
        
        self.save_meta()
        self.save_terrain()
        
        
    def save_meta(self):
        with open(os.path.join(self.output_dir, self.data['filename'], 'meta.txt'), 'w') as f:
            f.write(f"map_name: {self.data['map_name']}\n")
            f.write(f"width: {self.data['width']}\n")
            f.write(f"height: {self.data['height']}\n")
            f.write(f"game_length: {self.data['game_length']}\n")
            f.write(f"resolution_frame: {self.data['resolution_frame']}\n")
            for k, v in self.data['players_data'].items():
                if k == 'neutral': continue
                f.write(f"{k}: {v['name']}\n")
    
    def save_terrain(self):
        np.save(os.path.join(self.data['temp_dir'], 'terrain.npy'), self.data['terrain'])
        
    def _open_file_with_fallback(self, file_path):
        """
        Tries to open a file without specifying encoding. 
        If it fails, retries with cp949 encoding.
        """
        try:
            with open(file_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f:
                return [line.strip() for line in f.readlines()]

    def load_meta(self, path):
        print('Load meta...', end='')
        
        meta = self._open_file_with_fallback(os.path.join(path, 'meta'))
        meta = dict(zip(meta[0].split(','), meta[1].split(',')))
        
        meta['width'] = int(meta['width'])
        meta['height'] = int(meta['height'])
        meta['game_length'] = int(meta['game_length'])
        
        if 'players_data' in meta:
            players_list = [dict(zip(['name', 'race', 'start_location'], match.split(';')))
                            for match in re.findall(r'\[([^]]+)\]', meta['players_data'])]

            meta['players_data'] = {f"player_{i+1}": player for i, player in enumerate(players_list)}
            meta['players_data']['neutral'] = {'name': 'Neutral', 'race': None, 'start_location': None}

        print('Done')
        return meta

    def load_terrain(self, path, meta):
        print('Load terrain...', end='')
        
        terrain = self._open_file_with_fallback(os.path.join(path, 'terrain'))
        terrain = np.array([np.asarray(line.split(','), dtype=int) for line in terrain])
    
        print('Done')
        return terrain


    def load_vision(self, path, meta):
        print('Load vision...', end='')
        
        vision = self._open_file_with_fallback(os.path.join(path, 'vision'))
        
        print('Done')
        return vision


    def load_event(self, path):
        print('Load event...', end='')
        # first line is frame,x,y,player,unit,event
        events = self._open_file_with_fallback(os.path.join(path, 'event'))
        events = [
            {
                'frame': frame, 
                'x': x, 
                'y': y, 
                'player': player, 
                'unit_type': unit_type, 
                'event_type': event_type
            }
            for event in tqdm.tqdm(events[1:], desc='Load event...')
            for frame, x, y, player, unit_type, event_type in [event.split(',')]
            for frame, x, y in [map(int, [frame, x, y])]
        ]
        
        return events

    def load_state(self, path):
        print('Load state...', end='')
        state = self._open_file_with_fallback(os.path.join(path, 'state'))

        state_data = [line.split(',') for line in tqdm.tqdm(state[1:], desc='Load state...')]
        state_array = np.array(state_data)
        del state_data

        dataframe = pd.DataFrame(state_array, columns=[
            'frame', 'player', 'race', 'player_color', 'name', 'ID', 'x', 'y',
            'top', 'bottom', 'left', 'right', 'HP', 'max_HP', 'shield', 'max_shield',
            'energy', 'max_energy'
        ])

        int_columns = ['frame', 'x', 'y', 'top', 'bottom', 'left', 'right', 'HP',  'max_HP', 'shield', 'max_shield', 'energy', 'max_energy']
        dataframe[int_columns] = dataframe[int_columns].astype(int)

        filtered_df = dataframe.query("player != 'Neutral'")
        players_in_frame = filtered_df.groupby('frame')['player'].nunique().reset_index()

        resolution_frame = players_in_frame.loc[players_in_frame['player'] < 2, 'frame'].min()

        return resolution_frame, dataframe.to_dict('records')

    def load_data(self, file_path, args):
        meta = self.load_meta(file_path)
        resolution_frame, state_raw = self.load_state(file_path)
        
        if np.isnan(resolution_frame):
            resolution_frame = meta['game_length']
        
        return {
            'map_name': meta['map_name'],
            'width': meta['width'],
            'height': meta['height'],
            'game_length': meta['game_length'],
            'players_data': meta['players_data'],
            'terrain': self.load_terrain(file_path, meta),
            'vision_raw': self.load_vision(file_path, meta),
            # 'event_raw': self.load_event(file_path),
            'state_raw': state_raw,
            'resolution_frame': resolution_frame
        }
