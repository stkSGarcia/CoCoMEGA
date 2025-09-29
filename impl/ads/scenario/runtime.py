import os
import pickle
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from multiprocessing import Process, Queue

from impl.ads.utils.dpp_streaming import DPPStreamingSelector
from impl.ads.utils.leaderboad_utils import rulefit_vectorize, vectorize_realtime_data

from impl import config as cfg

logger = logging.getLogger(__name__)


class RuntimeScenarioManager:
    """
    Runtime Scenario Manager:
    - Loads all previously observed scenarios from a directory (each file is one scenario).
    - Maintains an in-memory vector cache for novelty checks.
    - Runs novelty checking & saving in a separate worker process.
    - supports both Cosine Max and Streaming DPP.
    """

    def __init__(self, dataset_dir: str, strategy: str):
        """
        :param dataset_dir: Directory containing per-scenario files.
        """
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

        if strategy in ("cosine_max", "dpp_stream",):
            self.strategy = strategy
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.angle_threshold_deg = cfg.CONFIG["runtime"]["novelty_detection"].get("angle_threshold_deg", 20.0)
        self.cosine_similarity_threshold = float(np.cos(np.deg2rad(self.angle_threshold_deg)))

        self.dpp_jitter = 1e-6
        self.dpp_max_selected = cfg.CONFIG["runtime"]["novelty_detection"].get("dpp_max_selected", None)

        # IPC
        self.queue: Queue = Queue()
        self.process: Optional[Process] = None

        # State (inside worker only)
        self._entries: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None
        self.counter: int = 0

    # ---------------- Public API ----------------

    def start(self):
        """Start worker process."""
        self.process = Process(target=self._worker, args=(self.queue,))
        self.process.start()
        logger.info("[RuntimeScenarioManager] Worker started (pid=%s)", self.process.pid)

    def submit(self, tick_data: Dict[str, Any], meta: Dict[str, Any]):
        """Submit new scenario for novelty check asynchronously."""
        if self.process is None:
            raise RuntimeError("RuntimeScenarioManager not started")
        self.queue.put((tick_data, meta))

    def stop(self):
        """Stop worker process gracefully."""
        if self.process is not None:
            self.queue.put(None)  # sentinel
            self.process.join()
            self.process = None
            logger.info("[RuntimeScenarioManager] Worker stopped")

    # ---------------- Worker logic ----------------

    def _worker(self, queue: Queue):
        """Worker loop: load existing, then process new submissions."""
        self._load_existing()

        # Initialize selector if using DPP
        if self.strategy == "dpp_stream":
            self._dpp_selector = DPPStreamingSelector(
                angle_threshold_deg=self.angle_threshold_deg,
                jitter=self.dpp_jitter,
                max_selected=self.dpp_max_selected
            )
            # Bootstrap with existing feats
            if self._matrix is not None and self._matrix.size > 0:
                self._dpp_selector.bootstrap_with(self._matrix)

        while True:
            msg = queue.get()
            if msg is None:  # stop signal
                logger.info("[RuntimeScenarioManager] Shutdown signal received")
                break

            tick_data, meta = msg
            try:
                scenario = vectorize_realtime_data(tick_data)
                vec = rulefit_vectorize(scenario)
                feat = self._l2_normalize(vec)

                # Novelty check
                logger.info("[RuntimeScenarioManager] Processing new scenario, feature norm: %.4f",
                            np.linalg.norm(feat))
                if self.strategy == "dpp_stream":
                    is_new = self._dpp_selector.consider(feat)
                    logger.info("[RuntimeScenarioManager] DPP selected: %s", is_new)
                elif self.strategy == "cosine_max":
                    max_sim, best_idx = self._max_cosine_similarity(feat)
                    logger.info("[RuntimeScenarioManager] Max similarity: %.4f (idx=%s)", max_sim, best_idx)
                    is_new = (max_sim <= self.cosine_similarity_threshold)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

                if is_new:
                    self.counter += 1
                    fname = os.path.join(self.dataset_dir, f"scenario_{self.counter:05d}.pkl")
                    print(fname)
                    with open(fname, "wb") as f:
                        pickle.dump(
                            {
                                "scenario": scenario,
                                "vector": vec,
                                "rt_data": tick_data,
                                "meta": meta,
                            },
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    self._entries.append({"vector": vec, "meta": meta})
                    if self._matrix is None or self._matrix.size == 0:
                        self._matrix = feat.reshape(1, -1)
                    else:
                        self._matrix = np.vstack([self._matrix, feat.reshape(1, -1)])
                    logger.info("[RuntimeScenarioManager] New scenario saved: %s", fname)
            except Exception as e:
                logger.warning("[RuntimeScenarioManager] Error processing scenario: %s", e)

    # ---------------- Internals ----------------

    def _load_existing(self):
        """Load all scenarios already saved in dataset_dir."""
        files = sorted(f for f in os.listdir(self.dataset_dir) if f.endswith(".pkl"))
        feats = []
        for f in files:
            path = os.path.join(self.dataset_dir, f)
            try:
                with open(path, "rb") as fh:
                    data = pickle.load(fh)
                vec = np.asarray(data["vector"], dtype=np.float32)
                feats.append(self._l2_normalize(vec))
                self._entries.append({"vector": vec, "meta": data.get("meta", {})})
                self.counter += 1
            except Exception as e:
                logger.warning("[RuntimeScenarioManager] Failed to load %s: %s", path, e)
        self._matrix = np.stack(feats, axis=0).astype(np.float32) if feats else None
        logger.info("[RuntimeScenarioManager] Loaded %d existing scenarios", len(self._entries))

    def _max_cosine_similarity(self, feat: np.ndarray) -> Tuple[float, Optional[int]]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return -1.0, None
        sims = (self._matrix @ feat.reshape(-1, 1)).reshape(-1)
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), best_idx

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < eps:
            return np.zeros((1,), dtype=np.float32)
        return v / n
