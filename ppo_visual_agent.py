from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO


class VisualPPOAgent:
    """Inference wrapper for PPO models trained by ppo_visual_trainer.py."""

    def __init__(
        self,
        model_path: str = "models/ppo_warlords_visual_third_0.zip",
        deterministic: bool = True,
        frame_size: int | None = None,
        stack_size: int | None = None,
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.model = PPO.load(str(self.model_path))
        self.deterministic = deterministic

        metadata_path = self.model_path.with_suffix(".metadata.json")
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        self.frame_size = int(frame_size or metadata.get("frame_size", 84))
        self.stack_size = int(stack_size or metadata.get("stack_size", 4))
        self.controlled_agent = metadata.get("controlled_agent")
        self.frames: deque[np.ndarray] = deque(maxlen=self.stack_size)

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        path = Path(model_path)
        candidates = [path]
        if path.suffix != ".zip":
            candidates.append(path.with_suffix(".zip"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Model not found. Tried: {', '.join(str(c) for c in candidates)}")

    def reset(self):
        self.frames.clear()

    def _preprocess_frame(self, obs) -> np.ndarray:
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = arr.astype(np.uint8, copy=False)
        arr = cv2.resize(arr, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        return arr

    def _stack_obs(self, obs) -> np.ndarray:
        frame = self._preprocess_frame(obs)

        if len(self.frames) == 0:
            for _ in range(self.stack_size):
                self.frames.append(frame.copy())
        else:
            self.frames.append(frame.copy())

        stacked = np.stack(list(self.frames), axis=0).astype(np.uint8)

        expected_shape = self.model.observation_space.shape
        if stacked.shape != expected_shape:
            raise ValueError(
                f"Observation shape mismatch. Got {stacked.shape}, "
                f"but model expects {expected_shape}."
            )

        return stacked

    def act(self, observation):
        obs = self._stack_obs(observation)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(np.asarray(action).item())
