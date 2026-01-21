"""Custom Gymnasium environment for signal weight optimization."""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SignalWeightingEnv(gym.Env):
    """Environment where agent learns to weight trading signals.

    Enhanced with: richer observations, reward normalization, transaction costs.
    """
    def __init__(self, data_path: str = "data/synthetic_data.csv", episode_length: int = 252,
                 lookback: int = 20, transaction_cost: float = 0.001):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.signal_cols = [c for c in self.data.columns if c.startswith("signal_")]
        self.n_signals = len(self.signal_cols)
        self.episode_length = episode_length
        self.lookback = lookback
        self.transaction_cost = transaction_cost

        self.prices = self.data["price"].values
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.signals = self.data[self.signal_cols].values

        # Observation: current signals + rolling mean + rolling std + momentum
        # = n_signals * 4 features
        obs_dim = self.n_signals * 4
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(0, 1, (self.n_signals,), np.float32)

        self.current_step = self.start_idx = 0
        self.prev_position = 0.0
        self.prev_weights = np.ones(self.n_signals) / self.n_signals

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

    def _get_observation(self) -> np.ndarray:
        """Build rich observation with rolling statistics."""
        idx = self.start_idx + self.current_step
        start = max(0, idx - self.lookback)

        # Current signals
        current = self.signals[idx]

        # Rolling statistics
        window = self.signals[start:idx + 1]
        if len(window) > 1:
            roll_mean = window.mean(axis=0)
            roll_std = window.std(axis=0) + 1e-8
            # Momentum: current vs mean
            momentum = (current - roll_mean) / roll_std
        else:
            roll_mean = current
            roll_std = np.ones(self.n_signals)
            momentum = np.zeros(self.n_signals)

        # Normalize current signals
        current_norm = (current - roll_mean) / roll_std

        obs = np.concatenate([current_norm, roll_mean, roll_std, momentum])
        return obs.astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        max_start = len(self.returns) - self.episode_length - 1
        self.start_idx = max(self.lookback, self.np_random.integers(self.lookback, max(self.lookback + 1, max_start)))
        self.current_step = 0
        self.prev_position = 0.0
        self.prev_weights = np.ones(self.n_signals) / self.n_signals
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Normalize weights
        weights = np.clip(action, 1e-8, None)
        weights = weights / weights.sum()

        idx = self.start_idx + self.current_step
        weighted_signal = np.dot(weights, self.signals[idx])

        # Position with stronger signal response
        position = np.tanh(weighted_signal * 3.0)

        # Trading return
        price_return = self.returns[idx]
        trading_return = position * price_return

        # Transaction cost based on position change
        turnover = abs(position - self.prev_position)
        cost = turnover * self.transaction_cost
        net_return = trading_return - cost

        # Reward: scaled return with small penalty for extreme positions
        reward = net_return * 100

        # Update running reward stats for info
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_var += delta * (reward - self.reward_mean)

        self.prev_position = position
        self.prev_weights = weights
        self.current_step += 1
        terminated = self.current_step >= self.episode_length

        obs = np.zeros(self.n_signals * 4, np.float32) if terminated else self._get_observation()
        info = {
            "weights": weights,
            "position": position,
            "trading_return": trading_return,
            "price_return": price_return,
            "turnover": turnover,
        }
        return obs, reward, terminated, False, info
