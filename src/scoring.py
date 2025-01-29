"""
Core scoring and allocation mechanisms for Lido's capital allocation framework.
Implements the performance-based scoring system and allocation logic.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from src.models import CapitalManager, DaoParams
from src.utils import continuous_ema

def calculate_risk_free_rate(staking_apy_series: pd.Series, window: int) -> float:
    """
    Calculate 7-day exponential moving average of stETH base yield.
    
    @param staking_apy_series: Historical series of staking APY
    @param window: EMA window in hours (default 7 days)
    @return: Current risk-free rate as EMA
    """
    if len(staking_apy_series) == 0:
        return 0.0  # Default value when no data available
    
    # Use minimum of available data length and window
    effective_window = min(len(staking_apy_series), window)
    return staking_apy_series.ewm(span=effective_window, adjust=False).mean().iloc[-1]


def calculate_bid_floor(
    prev_realized_yields: List[float], 
    risk_free_rate: float, 
    dao_premium: float
) -> float:
    
    median_prev_realized_yield = 0
    if len(prev_realized_yields) == 0:
        median_prev_realized_yield = risk_free_rate + dao_premium
    
    return max(risk_free_rate + dao_premium, median_prev_realized_yield)


def calculate_score(manager: CapitalManager, dao_params: DaoParams) -> tuple[float, str]:
    """Calculate raw score before normalization."""
    
    # Penalty factor from previous slashing
    penalty_factor = dao_params.penalty if manager.to_penalise else 1
    
    # Calculate delta between promised and realized yield
    delta = abs(manager.promised_yield - manager.realized_yield)
    
    # Calculate score components
    yield_component = dao_params.alpha * manager.realized_yield
    risk_component = manager.loss_buffer
    predictability_component = dao_params.beta * delta
    
    str_score = f'''
    score calculation components:
    yield: {yield_component}
    risk: {risk_component}
    predictability: {predictability_component}
    reputation: {manager.reputation}
    '''    
    
    # Calculate raw score without reputation (will be applied after normalization)
    raw_score = (yield_component * risk_component * (1 + predictability_component)) / penalty_factor
    
    return raw_score, str_score


def normalize_scores(managers: List[CapitalManager]) -> None:
    """L1 normalize scores across all managers."""
    scores = np.array([m.score for m in managers])
    total = np.sum(np.abs(scores))
    if total > 0:  # Avoid division by zero
        normalized = scores / total
        # Update managers with normalized scores
        for manager, norm_score in zip(managers, normalized):
            manager.score = norm_score


def allocate_capital(managers: List[CapitalManager], total_capital: float) -> Dict[str, float]:
    """Allocate capital based on normalized scores."""
    # Scores are already normalized, just multiply by total capital
    allocations = {m.manager_id: m.score * total_capital for m in managers}
    return allocations


def calculate_ema_reputation(manager: CapitalManager, time_delta: float, averaging_window: float) -> None:
    # Update reputation using continuous exponential moving average.
    
    # if no history, leave reputation alone (as 1.0)
    if not manager.score_history:
        return
    
    # Get latest score and previous EMA value
    current_score = manager.score_history.scores[-1]
    last_ema = manager.reputation if manager.reputation != 0 else current_score
    
    return continuous_ema(current_score, last_ema, time_delta, averaging_window)
