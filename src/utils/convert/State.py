import os
import pickle
import numpy as np
import pandas as pd
import tqdm
from multiprocessing import Process, Queue, cpu_count, Manager

from ..starcraft import UnitType, Category

# ==== 설정 ====
BINARY_FILL = False  # True면 채널 값을 0/1로(존재만), False면 타일 내 유닛 수 누적
# ===============

def _clip_rect(lt, rt, tp, bt, Wt, Ht):
    lt = max(0, min(int(lt), Wt - 1))
    rt = max(0, min(int(rt), Wt - 1))
    tp = max(0, min(int(tp), Ht - 1))
    bt = max(0, min(int(bt), Ht - 1))
    if lt > rt or tp > bt:
        return None
    return lt, rt, tp, bt

def worker(queue, temp_dir, progress_queue, height, width, players):
    """
    height/width는 '타일 기준'이어야 합니다 (예: mapHeightTiles, mapWidthTiles).
    players: {"1": {"name": ...}, "2": {"name": ...}, "neutral": {...}} 형태에서
             __init__에서 {player_name -> "1"/"2"/"neutral"}로 변환해 들어옵니다.
    """

    # 이름 -> UnitType 빠른 매핑 캐시
    unit_map = {name: ut for name, ut in UnitType.__members__.items()}

    # 카테고리 순서 고정
    PLAY_CATS = [Category.WORKER, Category.GROUND, Category.AIR, Category.BUILDING]
    NEU_CATS  = [Category.RESOURCE]

    def fill_rect(mat, lt, rt, tp, bt):
        if BINARY_FILL:
            mat[tp:bt+1, lt:rt+1] = 1
        else:
            mat[tp:bt+1, lt:rt+1] += 1

    def generate_player_stack(df, frame, player_id):
        """
        df: 해당 플레이어 행만 있는 DataFrame
        player_id: "1" / "2" / "neutral"
        저장 파일:
          - state_{frame}_player_1.npy  shape=(4,H,W)
          - state_{frame}_player_2.npy  shape=(4,H,W)
          - state_{frame}_neutral.npy   shape=(1,H,W)
        """
        if player_id == "neutral":
            stack = np.zeros((1, height, width), dtype=np.int32)
            if df.empty:
                np.save(os.path.join(temp_dir, f"state_{frame}_neutral.npy"), stack)
                return

            # 유닛 타입 해석
            ut = df["name"].map(lambda n: unit_map.get(n))
            valid = ut.notna()
            if not valid.any():
                np.save(os.path.join(temp_dir, f"state_{frame}_neutral.npy"), stack)
                return
            ut = ut[valid]
            sub = df.loc[valid]

            # RESOURCE 마스크
            is_res = ut.map(lambda u: u.belongs_to(Category.RESOURCE)).to_numpy()

            for idx, row in sub[is_res].iterrows():
                rect = _clip_rect(row["left_tile"], row["right_tile"]-1, row["top_tile"], row["bottom_tile"]-1, width, height)
                if rect is None:
                    continue
                lt, rt, tp, bt = rect
                fill_rect(stack[0], lt, rt, tp, bt)

            np.save(os.path.join(temp_dir, f"state_{frame}_neutral.npy"), stack)
            return

        # 플레이어 1/2: 4채널
        stack = np.zeros((4, height, width), dtype=np.int32)
        if df.empty:
            np.save(os.path.join(temp_dir, f"state_{frame}_player_{player_id}.npy"), stack)
            return

        ut = df["name"].map(lambda n: unit_map.get(n))
        valid = ut.notna()
        if not valid.any():
            np.save(os.path.join(temp_dir, f"state_{frame}_player_{player_id}.npy"), stack)
            return
        ut = ut[valid]
        sub = df.loc[valid]

        # 카테고리 마스크들
        masks = {
            Category.WORKER  : ut.map(lambda u: u.belongs_to(Category.WORKER)).to_numpy(),
            Category.GROUND  : ut.map(lambda u: (u.belongs_to(Category.GROUND)  and not u.belongs_to(Category.WORKER, Category.TRIVIAL))).to_numpy(),
            Category.AIR     : ut.map(lambda u: (u.belongs_to(Category.AIR)     and not u.belongs_to(Category.WORKER, Category.TRIVIAL))).to_numpy(),
            Category.BUILDING: ut.map(lambda u: (u.belongs_to(Category.BUILDING) and not u.belongs_to(Category.ADDON))).to_numpy(),
        }

        for ch, cat in enumerate(PLAY_CATS):
            m = masks[cat]
            if not m.any():
                continue
            for idx, row in sub[m].iterrows():
                rect = _clip_rect(row["left_tile"], row["right_tile"]-1, row["top_tile"], row["bottom_tile"]-1, width, height)
                if rect is None:
                    continue
                lt, rt, tp, bt = rect
                fill_rect(stack[ch], lt, rt, tp, bt)

        np.save(os.path.join(temp_dir, f"state_{frame}_player_{player_id}.npy"), stack)

    while True:
        frame = queue.get()
        if frame is None:
            break

        frame_file = os.path.join(temp_dir, f"state_{frame}.pkl")
        if not os.path.exists(frame_file):
            progress_queue.put(1)
            continue

        with open(frame_file, "rb") as f:
            # 프레임 테이블 로드
            frame_df = pd.DataFrame(pickle.load(f))

        # 필수 컬럼 확인
        need_cols = {"player","name","left_tile","right_tile","top_tile","bottom_tile"}
        missing = need_cols - set(frame_df.columns)
        if missing:
            raise KeyError(f"State frame {frame}: missing columns {missing}")

        # 플레이어 그룹핑
        for player_name, player_df in frame_df.groupby("player"):
            key = player_name.strip()
            pid = players.get(key)
            if pid is None:
                # 미지정 플레이어는 스킵(혹은 규칙을 정해 1/2로 분배)
                continue
            generate_player_stack(player_df, frame, pid)

        # pkl은 더이상 필요 없으면 정리
        os.remove(frame_file)
        progress_queue.put(1)


class State:
    def __init__(self, data):
        """
        data:
          - temp_dir: str
          - height: int (tiles)
          - width: int  (tiles)
          - resolution_frame: int
          - players_data: {
               "1": {"name": "PlayerA"}, "2": {"name": "PlayerB"},
               "neutral": {"name": "Neutral"}
             }
        """
        self.data = data
        self.temp_dir = data["temp_dir"]
        self.queue = Queue()
        self.num_workers = max(1, cpu_count() // 2)
        self.workers = []
        # 이름 -> "1"/"2"/"neutral"
        self.players = {v["name"].strip(): k for k, v in data["players_data"].items()}

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_value, traceback): pass

    def start_workers(self, progress_queue):
        Ht, Wt = self.data["height"], self.data["width"]
        self.workers = [
            Process(target=worker, args=(self.queue, self.temp_dir, progress_queue, Ht, Wt, self.players))
            for _ in range(self.num_workers)
        ]
        for p in self.workers:
            p.start()

    def stop_workers(self):
        for _ in self.workers:
            self.queue.put(None)
        for p in self.workers:
            p.join()

    def process(self, interval=1):
        frames = list(range(0, self.data["resolution_frame"], interval))
        with Manager() as manager:
            progress_queue = manager.Queue()
            self.start_workers(progress_queue)

            progress_bar = tqdm.tqdm(
                total=len(frames),
                desc="Converting State (channelized)...",
                miniters=max(1, len(frames) // 20)
            )

            for frame in frames:
                self.queue.put(frame)
            for _ in frames:
                progress_queue.get()
                progress_bar.update(1)
            progress_bar.close()

            self.stop_workers()

    def run(self, interval=1):
        self.process(interval=interval)
