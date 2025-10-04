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
    Build a {frame -> path} index from a glob pattern and a filename regex.

    Parameters
    ----------
    pattern : str
        Glob pattern (e.g., ".../state_*_player_1.npy").
    regex : str
        Regex to extract the frame number from filenames
        (e.g., r"state_(\d+)_player_1\.npy$").

    Returns
    -------
    Dict[int, str]
        Mapping from frame index to file path.
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
    Infer (H, W) in the following priority: terrain -> vision -> state (players).

    Returns
    -------
    (int, int)
        Inferred (H, W).

    Raises
    ------
    ValueError
        If the shape cannot be inferred from any provided source.
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
            # players: (4, H, W), neutral: (1, H, W)
            if arr.ndim == 3:
                return int(arr.shape[1]), int(arr.shape[2])

    raise ValueError("Failed to infer (H, W). Provide at least one of terrain/vision/state files.")


class Resolver:
    """
    Frame-driven resolver.

    Parameters
    ----------
    temp_dir : str
        Temporary directory that holds intermediate per-frame arrays.
    resolution_frame : int
        Number of output frames (processed for frames [0, resolution_frame-1]).
    output_dir : str
        Destination directory for resolved per-frame outputs.
    include_components : list[str], optional
        Any subset of:
          ["worker", "ground", "air", "building", "neutral", "vision", "terrain"].
        - worker/ground/air/building: filters from player channels and preserves this order
        - neutral: adds the neutral (resource) channel
        - vision : adds the vision channel (densified via carry-forward)
        - terrain: adds the static terrain channel
    height, width : int, optional
        If omitted, inferred from available inputs via `_infer_shape`.
    delete_inputs : bool
        If True, deletes consumed inputs after completion.

    Notes
    -----
    - Player arrays are expected as (4, H, W) in the order (worker, ground, air, building).
    - Neutral arrays are expected as (1, H, W).
    - Vision is sparse across frames and is densified by carrying forward the last seen frame.
    - Terrain is static and expected as (H, W).
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

        # Index inputs by frame
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

        # Infer H, W if needed
        if self.H is None or self.W is None:
            self.H, self.W = _infer_shape(
                self.terrain_path, self.vision_index, self.p1_index, self.p2_index, self.neu_index
            )

        # Pre-load terrain if requested
        self.terrain = None
        if self.include_terrain and self.terrain_path and os.path.exists(self.terrain_path):
            t = np.load(self.terrain_path)
            if t.ndim == 2:
                self.terrain = t
            else:
                raise ValueError("terrain.npy must be 2D (H, W)")

        # Pre-compute indices for player-type filtering
        self.type_indices = [self.UNIT_TYPE_INDEX[t] for t in self.want_types]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _load_player(self, idx: Dict[int, str], frame: int, name: str) -> np.ndarray:
        """
        Load a player tensor for a given frame; returns a zero (4, H, W) tensor if missing.

        Parameters
        ----------
        idx : Dict[int, str]
            Frame-to-path index for a player.
        frame : int
            Target frame.
        name : str
            Name used for error messages.

        Returns
        -------
        np.ndarray
            A (4, H, W) tensor.
        """
        if frame in idx and os.path.exists(idx[frame]):
            arr = np.load(idx[frame])
            # Expected: (4, H, W)
            if arr.ndim != 3 or arr.shape[1:] != (self.H, self.W):
                raise ValueError(f"{name} shape mismatch: {arr.shape}, expected (C, {self.H}, {self.W})")
            return arr
        return np.zeros((4, self.H, self.W), dtype=np.int32)

    def _load_neutral(self, frame: int) -> np.ndarray:
        """
        Load the neutral tensor for a given frame; returns a zero (1, H, W) tensor if missing.

        Returns
        -------
        np.ndarray
            A (1, H, W) tensor.
        """
        if frame in self.neu_index and os.path.exists(self.neu_index[frame]):
            arr = np.load(self.neu_index[frame])
            if arr.ndim != 3 or arr.shape[1:] != (self.H, self.W) or arr.shape[0] != 1:
                raise ValueError(f"neutral shape mismatch: {arr.shape}, expected (1, {self.H}, {self.W})")
            return arr
        return np.zeros((1, self.H, self.W), dtype=np.int32)

    def _load_vision_carry(self, last: Optional[np.ndarray], frame: int) -> np.ndarray:
        """
        Densify sparse vision by carry-forward.

        Parameters
        ----------
        last : Optional[np.ndarray]
            Previously carried (H, W) vision array, or None for initialization.
        frame : int
            Target frame.

        Returns
        -------
        np.ndarray
            A (1, H, W) tensor representing the current (carried) vision.
        """
        if not self.include_vision:
            return None
        if frame in self.vision_index and os.path.exists(self.vision_index[frame]):
            arr = np.load(self.vision_index[frame])
            if arr.ndim != 2 or arr.shape != (self.H, self.W):
                raise ValueError(f"vision[{frame}] shape mismatch: {arr.shape}, expected ({self.H}, {self.W})")
            last = arr
        if last is None:
            last = np.zeros((self.H, self.W), dtype=np.int32)
        return last[np.newaxis, ...]

    def _filter_types(self, player_arr: np.ndarray) -> np.ndarray:
        """
        Filter selected unit-type channels from a player tensor (4, H, W).

        Returns
        -------
        np.ndarray
            A (len(selected_types), H, W) tensor, possibly empty if no types requested.
        """
        if not self.type_indices:
            return np.empty((0, self.H, self.W), dtype=player_arr.dtype)
        return player_arr[self.type_indices, :, :]

    def run(self):
        """
        Resolve per-frame outputs up to `resolution_frame` and save as {frame}.npy in `output_dir`.
        Optionally removes consumed inputs if `delete_inputs=True`.
        """
        last_vision = None

        for f in tqdm.tqdm(
            range(self.resolution_frame),
            desc="Resolving outputs",
            miniters=max(1, self.resolution_frame // 20)
        ):
            out_list = []

            # Players
            p1 = self._load_player(self.p1_index, f, "player_1")
            p2 = self._load_player(self.p2_index, f, "player_2")

            # Filter by requested components
            p1 = self._filter_types(p1)
            p2 = self._filter_types(p2)

            out_list.append(p1)
            out_list.append(p2)

            # Neutral
            if self.include_neutral:
                neu = self._load_neutral(f)
                out_list.append(neu)

            # Vision (carry-forward)
            if self.include_vision:
                v = self._load_vision_carry(last_vision, f)
                last_vision = v[0]  # store as (H, W)
                out_list.append(v)

            # Terrain
            if self.include_terrain:
                if self.terrain is None:
                    # If absent, use zeros
                    out_list.append(np.zeros((1, self.H, self.W), dtype=np.int32))
                else:
                    out_list.append(self.terrain[np.newaxis, ...])

            # Stack and save
            result = np.vstack(out_list) if out_list else np.zeros((0, self.H, self.W), dtype=np.int32)
            np.save(os.path.join(self.output_dir, f"{f}.npy"), result)

        # Optional cleanup
        if self.delete_inputs:
            for d in (self.p1_index, self.p2_index, self.neu_index):
                for p in d.values():
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except:
                            pass
            if self.include_vision:
                for p in self.vision_index.values():
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except:
                            pass
            if self.terrain_path and os.path.exists(self.terrain_path):
                try:
                    os.remove(self.terrain_path)
                except:
                    pass
