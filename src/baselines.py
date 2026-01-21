"""Baseline (non-RL) strategies for comparison."""
import numpy as np
from abc import ABC, abstractmethod


class BaselineStrategy(ABC):
    """Abstract base class for baseline strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_weights(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        """Return signal weights given current signals and optional context."""
        pass

    def reset(self):
        """Reset any internal state. Override if needed."""
        pass


class EqualWeightStrategy(BaselineStrategy):
    """Uniform weighting across all signals."""

    @property
    def name(self) -> str:
        return "Equal Weight"

    def get_weights(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        n = len(signals)
        return np.ones(n) / n


class RandomWeightStrategy(BaselineStrategy):
    """Random weights (re-sampled each step)."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "Random Weight"

    def get_weights(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        w = self.rng.random(len(signals))
        return w / w.sum()


class BestSingleSignalStrategy(BaselineStrategy):
    """Use only the signal with highest absolute value (strongest conviction)."""

    @property
    def name(self) -> str:
        return "Best Single Signal"

    def get_weights(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        weights = np.zeros(len(signals))
        best_idx = np.argmax(np.abs(signals))
        weights[best_idx] = 1.0
        return weights


class MomentumWeightStrategy(BaselineStrategy):
    """Weight signals based on their recent performance (momentum)."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.signal_history: list[np.ndarray] = []
        self.return_history: list[float] = []

    @property
    def name(self) -> str:
        return "Momentum Weight"

    def reset(self):
        self.signal_history = []
        self.return_history = []

    def get_weights(self, signals: np.ndarray, price_return: float = 0.0, **kwargs) -> np.ndarray:
        self.signal_history.append(signals.copy())
        self.return_history.append(price_return)

        n = len(signals)
        if len(self.signal_history) < 5:
            return np.ones(n) / n

        # Compute correlation of each signal with returns over lookback
        window = min(self.lookback, len(self.signal_history) - 1)
        recent_signals = np.array(self.signal_history[-window-1:-1])
        recent_returns = np.array(self.return_history[-window:])

        correlations = np.zeros(n)
        for i in range(n):
            if recent_signals[:, i].std() > 1e-8:
                correlations[i] = np.corrcoef(recent_signals[:, i], recent_returns)[0, 1]

        # Convert to positive weights using softmax on correlations
        correlations = np.nan_to_num(correlations, 0)
        weights = np.exp(correlations * 2)
        return weights / weights.sum()


class InverseVolatilityStrategy(BaselineStrategy):
    """Weight signals inversely by their recent volatility."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.signal_history: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return "Inverse Volatility"

    def reset(self):
        self.signal_history = []

    def get_weights(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        self.signal_history.append(signals.copy())

        n = len(signals)
        if len(self.signal_history) < 5:
            return np.ones(n) / n

        window = min(self.lookback, len(self.signal_history))
        recent = np.array(self.signal_history[-window:])
        volatilities = recent.std(axis=0) + 1e-8

        weights = 1.0 / volatilities
        return weights / weights.sum()


class SignalStrengthStrategy(BaselineStrategy):
    """Weight proportional to signal magnitude (stronger signals get more weight)."""

    @property
    def name(self) -> str:
        return "Signal Strength"

    def get_weights(self, signals: np.ndarray, **kwargs) -> np.ndarray:
        # Use softmax on absolute signal values
        abs_signals = np.abs(signals) + 1e-8
        weights = np.exp(abs_signals)
        return weights / weights.sum()


def get_all_baselines(seed: int = 42) -> list[BaselineStrategy]:
    """Return all baseline strategies for comparison."""
    return [
        EqualWeightStrategy(),
        RandomWeightStrategy(seed=seed),
        BestSingleSignalStrategy(),
        MomentumWeightStrategy(lookback=20),
        InverseVolatilityStrategy(lookback=20),
        SignalStrengthStrategy(),
    ]
