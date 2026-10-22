from collections import defaultdict

import numpy as np


class ActionEnsembler:
    """Isolated copy of FastWAM's official temporal action ensembler."""

    def __init__(self):
        self.action_cache = defaultdict(list)

    def reset(self):
        self.action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int):
        if action_chunk.ndim == 3:
            action_chunk = action_chunk.squeeze(0)
        horizon, _ = action_chunk.shape
        for index in range(horizon):
            self.action_cache[start_timestamp + index].append(action_chunk[index, :])

    def get_action(self, timestamp: int) -> np.ndarray:
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")
        return np.mean(np.stack(self.action_cache[timestamp], axis=0), axis=0)

    def _cleanup(self, current_timestamp: int):
        for timestamp in [key for key in self.action_cache if key < current_timestamp]:
            del self.action_cache[timestamp]
