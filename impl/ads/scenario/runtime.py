import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from multiprocessing import Process, Queue
import time

from impl.ads.utils.dpp_streaming import DPPStreamingSelector
from impl.ads.utils.leaderboad_utils import rulefit_vectorize, vectorize_realtime_data

from impl import config as cfg
from impl.ads.utils.math_utils import block_normalize_rows

logger = logging.getLogger(__name__)


class RuntimeScenarioManager:
    """
    Runtime Scenario Manager:
    - Loads all previously observed scenarios from a directory (each file is one scenario).
    - Maintains an in-memory vector cache for novelty checks.
    - Runs novelty checking & saving in a separate worker process.
    - supports both Cosine Max and Streaming DPP.
    """

    def __init__(self, dataset_dir: str, strategy: str, discard_original_features: bool,
                 keep_zero_vectors: bool = False):
        """
        :param dataset_dir: Directory containing per-scenario files.
        :param strategy: Novelty detection strategy: "none", "cosine_max", "dpp_stream".
        :param discard_original_features: If True, only store the vectorized features, not the original scenario.
        :param keep_zero_vectors: If True, keep scenarios that vectorize to all zeros.
        """
        self.dataset_dir = dataset_dir
        if self.dataset_dir:
            os.makedirs(self.dataset_dir, exist_ok=True)

        if strategy in ("none", "cosine_max", "dpp_stream",):
            self.strategy = strategy
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        self.discard_original_features = discard_original_features
        self.keep_zero_vectors = keep_zero_vectors
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
        # Run-wise unique id to avoid filename collisions across runs
        self.run_id: str = f"{int(time.time() * 1000)}_{os.getpid()}"

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
                vec, block_sizes = rulefit_vectorize(scenario, keep_original=not self.discard_original_features)
                feat = np.asarray(vec, dtype=np.float32)

                # check if vec is not all zeros
                if np.all(vec == 0):
                    logger.warning(f"[RuntimeScenarioManager] All zeros vector found for scenario {fname}")
                    if self.keep_zero_vectors:
                        is_new = True
                    else:
                        is_new = False
                else:
                    feat = block_normalize_rows(vec, block_sizes)
                    # Novelty check
                    if self.strategy == "dpp_stream":
                        is_new, d2, angle_deg = self._dpp_selector.consider(feat)
                        if is_new and d2 is None:
                            logger.info(
                                f"[RuntimeScenarioManager] DPP selected: {is_new}, as there are no selected scenarios")
                        elif not is_new and d2 is None:
                            logger.info(f"[RuntimeScenarioManager] DPP selected: {is_new}, max selected reached")
                        else:
                            logger.info("[RuntimeScenarioManager] DPP selected: %s, d2: %.4f, angle_deg: %.4f", is_new,
                                        d2, angle_deg)
                    elif self.strategy == "cosine_max":
                        max_sim, best_idx = self._max_cosine_similarity(feat)
                        if max_sim is None and best_idx is None:
                            is_new = True
                            logger.info(
                                f"[RuntimeScenarioManager] Max cosine selected: {is_new}, as there are no selected scenarios")
                        else:
                            angle_deg = np.degrees(np.arccos(max_sim))
                            is_new = (max_sim <= self.cosine_similarity_threshold)
                            logger.info(
                                f"[RuntimeScenarioManager] Max cosine selected: {is_new}, closest_idx={best_idx}, angle_deg: {angle_deg}")
                            logger.debug(f"[RuntimeScenarioManager] Max cosine similarity: {max_sim}, feature: {feat}")

                    elif self.strategy == "none":
                        logger.info("[RuntimeScenarioManager] No strategy selected, scenario accepted")
                        is_new = True
                    else:
                        raise ValueError(f"Unknown strategy: {self.strategy}")

                if is_new:
                    self.counter += 1
                    if meta.get("save_path", None):
                        fname = os.path.join(self.dataset_dir, meta["save_path"],
                                             f"scenario_{self.run_id}_{self.counter:05d}.pkl")
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                    else:
                        fname = os.path.join(self.dataset_dir, f"scenario_{self.run_id}_{self.counter:05d}.pkl")
                    with open(fname, "wb") as f:
                        pickle.dump(
                            {
                                "scenario": scenario,
                                "vector": vec,
                                "vector_norm": feat,
                                "block_sizes": block_sizes,
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
                raise e

    # ---------------- Internals ----------------

    def _load_existing(self):
        """Load all scenarios already saved in dataset_dir."""
        files = sorted(Path(self.dataset_dir).rglob("*.pkl"))
        feats = []
        for f in files:
            path = os.path.join(self.dataset_dir, f)
            try:
                with open(path, "rb") as fh:
                    data = pickle.load(fh)
                vec = np.asarray(data["vector"], dtype=np.float64)
                vec_norm = np.asarray(data["vector_norm"], dtype=np.float64)

                feats.append(vec_norm.reshape(1, -1))
                self._entries.append({"vector": vec, "vector_norm": vec_norm, "meta": data.get("meta", {})})
                self.counter += 1
            except Exception as e:
                logger.warning("[RuntimeScenarioManager] Failed to load %s: %s", path, e)
        self._matrix = np.vstack(feats).astype(np.float64) if feats else None
        logger.info("[RuntimeScenarioManager] Loaded %d existing scenarios", len(self._entries))

    def _max_cosine_similarity(self, feat: np.ndarray) -> Tuple[float, Optional[int]]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return None, None
        sims = (self._matrix @ feat.reshape(-1, 1)).reshape(-1)
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), best_idx
