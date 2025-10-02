# resolver.py
import os
import re
import glob
import numpy as np
import tqdm
from typing import Dict, Tuple, Optional, List


def sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _index_by_frame(pattern: str, regex: str) -> Dict[int, str]:
    """
    pattern: glob 패턴 (예: ".../state_*_player_1.npy")
    regex  : 파일명에서 프레임 추출할 정규식 (예: r"state_(\d+)_player_1\.npy$")
    """
    out = {}
    for path in glob.glob(pattern):
        m = re.search(regex, os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def _infer_shape(
    terrain_path: Optional[str],
    vision_index: Dict[int, str],
    p1_index: Dict[int, str],
    p2_index: Dict[int, str],
    neu_index: Dict[int, str],
) -> Tuple[int, int]:
    """
    H, W 추론: terrain -> vision -> state(player) 순으로 시도
    """
    # terrain
    if terrain_path and os.path.exists(terrain_path):
        arr = np.load(terrain_path)
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])

    # vision
    if vision_index:
        arr = np.load(vision_index[sorted(vision_index.keys())[0]])
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])

    # player state
    for idx in (p1_index, p2_index, neu_index):
        if idx:
            arr = np.load(idx[sorted(idx.keys())[0]])
            # player: (4,H,W), neutral: (1,H,W)
            if arr.ndim == 3:
                return int(arr.shape[1]), int(arr.shape[2])

    raise ValueError("Failed to infer (H, W). Provide at least one of terrain/vision/state files.")


class Resolver:
    """
    프레임 주도형 리졸버.
    - resolution_frame: 출력 프레임 개수 (0..resolution_frame-1)
    - include_components: ["worker","ground","air","building","neutral","vision","terrain"]
      * worker/ground/air/building: 플레이어 채널에서 그 순서로 필터
      * neutral: 중립(자원) 채널 추가
      * vision : 시야 채널 추가 (carry-forward로 덴시파이)
      * terrain: 지형 채널(고정) 추가
    """

    UNIT_TYPE_INDEX = {"worker": 0, "ground": 1, "air": 2, "building": 3}

    def __init__(
        self,
        temp_dir: str,
        resolution_frame: int,
        output_dir: str,
        include_components: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        delete_inputs: bool = False,
    ):
        self.temp_dir = os.path.abspath(temp_dir)
        self.resolution_frame = int(resolution_frame)
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.include_components = include_components or [
            "worker", "ground", "air", "building", "neutral", "vision", "terrain"
        ]
        self.want_types = [k for k in self.UNIT_TYPE_INDEX if k in self.include_components]
        self.include_neutral = "neutral" in self.include_components
        self.include_vision = "vision" in self.include_components
        self.include_terrain = "terrain" in self.include_components

        self.H = height
        self.W = width
        self.delete_inputs = bool(delete_inputs)

        # index inputs by frame
        self.p1_index = _index_by_frame(
            os.path.join(self.temp_dir, "state_*_player_1.npy"),
            r"state_(\d+)_player_1\.npy$",
        )
        self.p2_index = _index_by_frame(
            os.path.join(self.temp_dir, "state_*_player_2.npy"),
            r"state_(\d+)_player_2\.npy$",
        )
        self.neu_index = _index_by_frame(
            os.path.join(self.temp_dir, "state_*_neutral.npy"),
            r"state_(\d+)_neutral\.npy$",
        )
        # vision is sparse
        self.vision_index = _index_by_frame(
            os.path.join(self.temp_dir, "vision_*.npy"),
            r"vision_(\d+)\.npy$",
        )
        # terrain (single file)
        terrain_files = glob.glob(os.path.join(self.temp_dir, "terrain.npy"))
        self.terrain_path = terrain_files[0] if (self.include_terrain and terrain_files) else None

        # infer shape if needed
        if self.H is None or self.W is None:
            self.H, self.W = _infer_shape(
                self.terrain_path, self.vision_index, self.p1_index, self.p2_index, self.neu_index
            )

        # pre-load terrain if used
        self.terrain = None
        if self.include_terrain and self.terrain_path and os.path.exists(self.terrain_path):
            t = np.load(self.terrain_path)
            if t.ndim == 2:
                self.terrain = t
            else:
                raise ValueError("terrain.npy must be 2D (H,W)")

        # pre-check type indices for filtering
        self.type_indices = [self.UNIT_TYPE_INDEX[t] for t in self.want_types]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _load_player(self, idx: Dict[int, str], frame: int, name: str) -> np.ndarray:
        """
        로드 실패 시 (4,H,W) 제로 채널 반환
        """
        if frame in idx and os.path.exists(idx[frame]):
            arr = np.load(idx[frame])
            # 기대: (4,H,W)
            if arr.ndim != 3 or arr.shape[1:] != (self.H, self.W):
                raise ValueError(f"{name} shape mismatch: {arr.shape}, expected (C,{self.H},{self.W})")
            return arr
        return np.zeros((4, self.H, self.W), dtype=np.int32)

    def _load_neutral(self, frame: int) -> np.ndarray:
        """
        로드 실패 시 (1,H,W) 제로 채널 반환
        """
        if frame in self.neu_index and os.path.exists(self.neu_index[frame]):
            arr = np.load(self.neu_index[frame])
            if arr.ndim != 3 or arr.shape[1:] != (self.H, self.W) or arr.shape[0] != 1:
                raise ValueError(f"neutral shape mismatch: {arr.shape}, expected (1,{self.H},{self.W})")
            return arr
        return np.zeros((1, self.H, self.W), dtype=np.int32)

    def _load_vision_carry(self, last: Optional[np.ndarray], frame: int) -> np.ndarray:
        """
        스파스 vision을 carry-forward로 덴시파이.
        반환: (1,H,W)
        """
        if not self.include_vision:
            return None
        if frame in self.vision_index and os.path.exists(self.vision_index[frame]):
            arr = np.load(self.vision_index[frame])
            if arr.ndim != 2 or arr.shape != (self.H, self.W):
                raise ValueError(f"vision[{frame}] shape mismatch: {arr.shape}, expected ({self.H},{self.W})")
            last = arr
        if last is None:
            last = np.zeros((self.H, self.W), dtype=np.int32)
        return last[np.newaxis, ...]

    def _filter_types(self, player_arr: np.ndarray) -> np.ndarray:
        """
        플레이어 채널(4,H,W)에서 선택된 타입만 추출.
        """
        if not self.type_indices:
            return np.empty((0, self.H, self.W), dtype=player_arr.dtype)
        return player_arr[self.type_indices, :, :]

    def run(self):
        last_vision = None

        for f in tqdm.tqdm(range(self.resolution_frame), desc="Resolving outputs", miniters=max(1, self.resolution_frame // 20)):
            out_list = []

            # players
            p1 = self._load_player(self.p1_index, f, "player_1")
            p2 = self._load_player(self.p2_index, f, "player_2")

            # filter by include_components
            p1 = self._filter_types(p1)
            p2 = self._filter_types(p2)

            out_list.append(p1)
            out_list.append(p2)

            # neutral
            if self.include_neutral:
                neu = self._load_neutral(f)
                out_list.append(neu)

            # vision (carry-forward)
            if self.include_vision:
                v = self._load_vision_carry(last_vision, f)
                last_vision = v[0]  # (1,H,W) -> (H,W) 저장
                out_list.append(v)

            # terrain
            if self.include_terrain:
                if self.terrain is None:
                    # 없으면 0으로
                    out_list.append(np.zeros((1, self.H, self.W), dtype=np.int32))
                else:
                    out_list.append(self.terrain[np.newaxis, ...])

            # stack and save
            result = np.vstack(out_list) if out_list else np.zeros((0, self.H, self.W), dtype=np.int32)
            np.save(os.path.join(self.output_dir, f"{f}.npy"), result)

        # 필요한 경우 입력 정리
        if self.delete_inputs:
            for d in (self.p1_index, self.p2_index, self.neu_index):
                for p in d.values():
                    if os.path.exists(p):
                        try: os.remove(p)
                        except: pass
            if self.include_vision:
                for p in self.vision_index.values():
                    if os.path.exists(p):
                        try: os.remove(p)
                        except: pass
            if self.terrain_path and os.path.exists(self.terrain_path):
                try: os.remove(self.terrain_path)
                except: pass
