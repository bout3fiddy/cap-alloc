"""
Data models for capital allocation simulation.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class ScoreHistory:
    """Tracks score history with timestamps."""
    timestamps: List[pd.Timestamp] = None
    scores: List[float] = None
    
    def __post_init__(self):
        self.timestamps = [] if self.timestamps is None else self.timestamps
        self.scores = [] if self.scores is None else self.scores
    
    def add(self, timestamp: pd.Timestamp, score: float):
        self.timestamps.append(timestamp)
        self.scores.append(score)
    
    def get_last_time_delta(self) -> float:
        """Get time delta in hours between last two scores."""
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[-2]).total_seconds() / 3600


@dataclass
class CapitalManager:
    """Represents a capital manager participating in the allocation system."""
    manager_id: str
    promised_yield: float
    realized_yield: float
    allocated_capital: float
    absolute_returns: float
    score: float
    str_score: str
    reputation: float
    score_history: ScoreHistory
    bond_amount: float
    to_penalise: bool
    strategy_risk: float
    
    def __post_init__(self):
        self.score_history = ScoreHistory() if self.score_history is None else self.score_history


@dataclass
class DaoParams:
    alpha: float                    # Yield component weight
    beta: float                     # Predictability component weight
    premium: float                  # Risk premium over risk-free rate
    min_bond: int                   # Minimum required LDO bond
    epoch_length_hours: int         # number of hours in an epoch
    auction_length: int             # number of hours before epoch start for submitting bids
    ema_window_risk_free: int     # EMA of risk free rate
    ema_window_reputation: int      # EMA of reputation score
    penalty: float                  # penalty to score for underperformance
    slash_percent: float            # slash to bond