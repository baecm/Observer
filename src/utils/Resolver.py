import os
import glob
import re
import numpy as np
import tqdm

def sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class Resolver:
    def __init__(self, temp_dir, resolution_frame, output_dir):
        self.temp_dir = temp_dir
        self.resolution_frame = resolution_frame
        self.output_dir = output_dir
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def run(self):
        temp_dir = os.path.abspath(self.temp_dir)
    
        state_player_1 = sorted(glob.glob(os.path.join(temp_dir, 'state_*_player_1.npy')), key=sort_key)
        state_player_2 = sorted(glob.glob(os.path.join(temp_dir, 'state_*_player_2.npy')), key=sort_key)
        state_neutral = sorted(glob.glob(os.path.join(temp_dir, 'state_*_neutral.npy')), key=sort_key)
        vision_files = sorted(glob.glob(os.path.join(temp_dir, 'vision_*.npy')), key=sort_key)
        terrain_files = glob.glob(os.path.join(temp_dir, 'terrain.npy'))
        
        for i in tqdm(range(self.resolution_frame), desc='Resolving outputs'):
            state_player_1 = np.load(state_player_1[i])
            state_player_2 = np.load(state_player_2[i])
            state_neutral = np.load(state_neutral[i])
            vision = np.load(vision_files[i])
            terrain = np.load(terrain_files[i])
            
            result = np.vstack((state_player_1,
                                state_player_2,
                                state_neutral,
                                np.expand_dim(vision, axis=0),
                                np.expand_dim(terrain, axis=0)))
            
            np.save(os.path.join(self.output_dir, f'{i}.npy'), result)
        