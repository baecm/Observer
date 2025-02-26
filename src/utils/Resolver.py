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

        state_player_1_files = sorted(glob.glob(os.path.join(temp_dir, 'state_*_player_1.npy')), key=sort_key)
        state_player_2_files = sorted(glob.glob(os.path.join(temp_dir, 'state_*_player_2.npy')), key=sort_key)
        state_neutral_files = sorted(glob.glob(os.path.join(temp_dir, 'state_*_neutral.npy')), key=sort_key)
        vision_files = sorted(glob.glob(os.path.join(temp_dir, 'vision_*.npy')), key=sort_key)
        terrain_files = glob.glob(os.path.join(temp_dir, 'terrain.npy'))

        if not terrain_files:
            raise FileNotFoundError("terrain.npy 파일이 존재하지 않습니다.")

        terrain = np.load(terrain_files[0])  # terrain 파일은 하나만 존재한다고 가정

        for i in tqdm.tqdm(range(self.resolution_frame), desc='Resolving outputs'):
            state_player_1 = np.load(state_player_1_files[i])
            state_player_2 = np.load(state_player_2_files[i])
            state_neutral = np.load(state_neutral_files[i])
            vision = np.load(vision_files[i])

            result = np.vstack((
                state_player_1,
                state_player_2,
                state_neutral,
                np.expand_dims(vision, axis=0),
                np.expand_dims(terrain, axis=0)
            ))

            os.remove(state_player_1_files[i])
            os.remove(state_player_2_files[i])
            os.remove(state_neutral_files[i])
            os.remove(vision_files[i])
            
            np.save(os.path.join(self.output_dir, f'{i}.npy'), result)
