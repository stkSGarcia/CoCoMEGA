import os
import pickle
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from multiprocessing import Process, Queue
import time

from impl.ads.utils.leaderboad_utils import rulefit_vectorize, vectorize_realtime_data

from impl import config as cfg
from impl.ads.utils.math_utils import block_normalize_rows, d2_to_angle_deg, solve_lower

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
                 keep_zero_vectors: bool = False, partitions=[]):
        """
        :param dataset_dir: Directory containing per-scenario files.
        :param strategy: Novelty detection strategy: "none", "cosine_max", "dpp_stream".
        :param discard_original_features: If True, only store the vectorized features, not the original scenario.
        :param keep_zero_vectors: If True, keep scenarios that vectorize to all zeros.
        :param partitions: List of features to partition when comparing scenarios.
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

        # IPC
        self.queue: Queue = Queue()
        self.process: Optional[Process] = None

        # State (inside worker only)
        self._entries: Dict[str, Any] = {}
        self._matrix: Dict[str, Any] = {}
        self.partitions = partitions
        self.n_sel: int = 0
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

    def consider(self, feat: np.ndarray, partition_values=[]):
        """Decide whether to accept a candidate feature (unit-normalized row)."""
        return True, {"reason": "Always accept in base class"}

    # ---------------- Worker logic ----------------

    def _worker(self, queue: Queue):
        """Worker loop: load existing, then process new submissions."""
        self._load_existing()

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
                partition_values = [getattr(scenario, k, "unknown") for k in self.partitions]

                # check if vec is not all zeros
                if np.all(vec == 0):
                    logger.warning(f"[RuntimeScenarioManager] All zeros vector found for scenario {fname}")
                    is_new = self.keep_zero_vectors
                else:
                    feat = block_normalize_rows(vec, block_sizes)
                    # Novelty check
                    is_new, info = self.consider(feat, partition_values)
                    logger.info(f"[RuntimeScenarioManager] Selected: {is_new}, info: {info}")

                if is_new:
                    if meta.get("save_path", None):
                        fname = os.path.join(self.dataset_dir, meta["save_path"],
                                             f"scenario_{self.run_id}_{self.n_sel:05d}.pkl")
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                    else:
                        fname = os.path.join(self.dataset_dir, f"scenario_{self.run_id}_{self.n_sel:05d}.pkl")
                    entry = {
                        "scenario": scenario,
                        "vector": vec,
                        "vector_norm": feat,
                        "block_sizes": block_sizes,
                        "rt_data": tick_data,
                        "meta": meta,
                    }
                    with open(fname, "wb") as f:
                        pickle.dump(
                            entry,
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    self._add_entry(entry)
                    logger.info("[RuntimeScenarioManager] New scenario saved: %s", fname)
            except Exception as e:
                logger.warning("[RuntimeScenarioManager] Error processing scenario: %s", e)
                raise e

    # ---------------- Internals ----------------

    def _load_existing(self):
        """Load all scenarios already saved in dataset_dir."""
        files = sorted(Path(self.dataset_dir).rglob("*.pkl"))
        for f in files:
            path = os.path.join(self.dataset_dir, f)
            try:
                with open(path, "rb") as fh:
                    data = pickle.load(fh)
                self._add_entry(data)
            except Exception as e:
                logger.warning("[RuntimeScenarioManager] Failed to load %s: %s", path, e)

        logger.info("[RuntimeScenarioManager] Loaded %d existing scenarios", self.n_sel)

    def _add_entry(self, entry: Dict[str, Any]):
        """Add entry to hierarchical entries based on partitions."""
        partition_values = self._get_partition_values(entry)
        entry_loc = self._get_loc(self._entries, partition_values)
        mat_loc = self._get_loc(self._matrix, partition_values)

        if "list" not in entry_loc:
            entry_loc["list"] = []
        entry_loc["list"].append(entry)

        # Also add to corresponding matrix
        vec_norm = entry["vector_norm"].reshape(1, -1)
        if "mat" not in mat_loc:
            mat_loc["mat"] = vec_norm
        else:
            mat_loc["mat"] = np.vstack([mat_loc["mat"], vec_norm])

        self.n_sel += 1

    def _get_loc(self, root: Dict[str, Any], partition_values: List[str]) -> Optional[
        Dict[str, Any]]:
        """Get location in hierarchical structure based on partition values."""
        loc = root
        for partition_key, partition_value in zip(self.partitions, partition_values):
            _partition = f"{partition_key}={partition_value.lower()}"
            if _partition not in loc:
                loc[_partition] = {}
            loc = loc[_partition]
        return loc

    def _get_partition_values(self, entry: Dict[str, Any]) -> List[str]:
        """Get partition values from an entry."""
        return [getattr(entry["scenario"], k, "unknown") for k in self.partitions]


class CosineMaxSelector(RuntimeScenarioManager):
    """
    Max Cosine Similarity Selector:
    - Loads all previously observed scenarios from a directory (each file is one scenario).
    - Maintains an in-memory vector cache for novelty checks.
    - Runs novelty checking & saving in a separate worker process.
    """

    def __init__(self, dataset_dir: str, discard_original_features: bool,
                 keep_zero_vectors: bool = False, partitions=[]):
        """
        :param dataset_dir: Directory containing per-scenario files.
        :param discard_original_features: If True, only store the vectorized features, not the original scenario.
        :param keep_zero_vectors: If True, keep scenarios that vectorize to all zeros.
        :param partitions: List of features to partition when comparing scenarios.
        """
        super().__init__(dataset_dir,
                         discard_original_features=discard_original_features,
                         keep_zero_vectors=keep_zero_vectors,
                         partitions=partitions)

        self.cosine_similarity_threshold = float(np.cos(np.deg2rad(self.angle_threshold_deg)))

    def consider(self, feat: np.ndarray, partition_values=[]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Compute max cosine similarity between feat and existing matrix.
        :param feat: Feature vector (unit-normalized).
        :return: (max_similarity, best_index) or (None, None) if no existing entries.
        """
        loc = self._get_loc(self._matrix, partition_values)
        if "mat" not in loc or loc["mat"] is None or loc["mat"].shape[0] == 0:
            return True, {"reason": "No existing entries in the selected partition"}

        sims = (loc["mat"] @ feat.reshape(-1, 1)).reshape(-1)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        angle_deg = np.degrees(np.arccos(max_sim))
        is_new = (max_sim <= self.cosine_similarity_threshold)
        return is_new, {"angle_deg": angle_deg, "max_sim": max_sim, "best_idx": best_idx},


class StreamingDPPSelector(RuntimeScenarioManager):
    """
    Streaming DPP Selector:
    - Loads all previously observed scenarios from a directory (each file is one scenario).
    - Maintains an in-memory vector cache for novelty checks.
    - Runs novelty checking & saving in a separate worker process.
    """

    def __init__(self, dataset_dir: str, discard_original_features: bool,
                 keep_zero_vectors: bool = False, partitions=[]):
        """
        :param dataset_dir: Directory containing per-scenario files.
        :param discard_original_features: If True, only store the vectorized features, not the original scenario.
        :param keep_zero_vectors: If True, keep scenarios that vectorize to all zeros.
        :param partitions: List of features to partition when comparing scenarios.
        """
        super().__init__(dataset_dir,
                         discard_original_features=discard_original_features,
                         keep_zero_vectors=keep_zero_vectors,
                         partitions=partitions)

        self.jitter = cfg.CONFIG["runtime"]["novelty_detection"].get("dpp_jitter", 1e-6)
        self.max_selected = cfg.CONFIG["runtime"]["novelty_detection"].get("dpp_max_selected", None)
        self.tau = float(np.sin(np.deg2rad(self.angle_threshold_deg)) ** 2)
        # State
        self.R: Optional[np.ndarray] = None  # upper-triangular Cholesky of K_SS

    def consider(self, feat: np.ndarray, partition_values=[]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Decide whether to accept a candidate feature using streaming DPP.
        """

        if self.max_selected is not None and self.n_sel >= self.max_selected:
            return False, {"reason": "Max selected reached"}

        loc = self._get_loc(self._matrix, partition_values)
        if ("mat" not in loc) or (loc["mat"] is None) or (loc["mat"].shape[0] == 0):
            return True, {"reason": "No existing entries in the selected partition"}

        # k_iS = X @ feat  (cosine kernel on unit vectors)
        k_iS = loc["mat"] @ feat
        k_ii = 1.0 + self.jitter

        # Solve R^T w = k_iS  (forward substitution on lower-triangular R^T)
        w = self._solve_lower(RT=loc["R"].T, b=k_iS)

        # Schur complement d^2
        d2 = float(k_ii - np.dot(w, w))
        angle_deg = d2_to_angle_deg(d2)
        return d2 >= self.tau, {"d2": d2, "angle_deg": angle_deg}

    def _add_entry(self, entry: Dict[str, Any]):
        """Add entry to hierarchical entries based on partitions."""

        partition_values = self._get_partition_values(entry)
        loc = self._get_loc(root=self._matrix, partition_values=partition_values)
        feat = entry["vector_norm"].reshape(-1, 1)
        if "R" not in loc:
            d = math.sqrt(1.0 + self.jitter)
            loc["R"] = np.array([[d]], dtype=np.float32)
        else:
            k_iS = loc["mat"] @ feat
            w = solve_lower(RT=loc["R"].T, b=k_iS)
            d2 = float(1.0 + self.jitter - np.dot(w, w))
            d = math.sqrt(max(d2, self.jitter))

            loc["R"] = np.block([
                [loc["R"], w[:, None]],
                [np.zeros((1, self.n_sel), dtype=np.float32), np.array([[d]], dtype=np.float32)]
            ])

        super()._add_entry(entry)
