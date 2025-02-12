"""
Simulation engine for capital allocation mechanism.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from src.models import CapitalManager, DaoParams
from src import scoring
from src.utils import get_group

class AllocationSimulator:
    """Simulates capital allocation over multiple epochs."""
    
    def __init__(
            self, 
            data: pd.DataFrame,
            initial_managers: List[CapitalManager],
            dao_params: DaoParams,
            initial_capital: float,
            strategy_liq_buffer: float
        ):
        self.resampled_data = data.resample(f'{dao_params.epoch_length_hours}h')
        self.epoch_length_hours = dao_params.epoch_length_hours
        self.managers = {m.manager_id: m for m in initial_managers}
        self.manager_ids = self.managers.keys()
        self.dao_params = dao_params
        self.capital = initial_capital
        self.previous_capital = initial_capital
        self.strategy_liq_buffer = strategy_liq_buffer
        
        self.history = []
        
        # Equally distribute initial capital across managers
        equal_allocation = initial_capital / len(self.managers)
        for manager in self.managers.values():
            manager.allocated_capital = equal_allocation
        
    def simulate_epoch(self, epoch_id: int) -> Dict:
        """Simulate a single epoch's dynamics."""
        
        print("$$$$$$$$$$$")
        print(f"epoch ID: {epoch_id}")
        
        # pick out grouped data from the epoch
        # the grouped data contains the ema risk free rate and
        # the net apy of the strategy from real world data (into the block)
        epoch_data = get_group(self.resampled_data, epoch_id)
        
        # for ema rate, we only care about the data available at auction 
        # start, which is in the dao params. note that this works for
        # hourly data for this simulation purpose
        ema_risk_free = epoch_data.ema_riskfree[epoch_data.index[self.dao_params.auction_length]]
        
        # Calculate bid floor of the epoch using last hour's data for auction
        bid_floor = scoring.calculate_bid_floor(
            prev_realized_yields=[m.realized_yield for m in self.managers.values()],
            risk_free_rate=ema_risk_free,
            dao_premium=self.dao_params.premium
        )
        
        capital_at_end = 0
        # we simulate health over the epoch and calculate scores for each manager
        for manager in self.managers.values():

            # for each manager, simulate health
            health_factors = self._simulate_position_health(
                manager, len(epoch_data), self.strategy_liq_buffer,
            )
            
            # Simulate promised yield based on manager's risk profile
            # Higher risk managers will deviate more from actual returns, but within reasonable bounds
            # Base deviation scales with risk level - riskier managers are more likely to over/under promise
            average_net_apy_realdata = epoch_data['net_apy'].mean()
            base_deviation = 0.03 + (0.07 * manager.strategy_risk)  # Maps 0.1->0.037, 0.9->0.093
            manager.promised_yield = average_net_apy_realdata * np.random.normal(1.0, base_deviation)
            
            # simulate supply apy referencing manager's risk
            # the risk component here is hypothetical to simulate returns in lieu of data from multiple sources
            
            # Adjust realized yield based on risk level - higher risk means slightly higher returns
            # For risk=0.1 (low), multiplier will be ~0.8 giving ~7% on 9% avg
            # For risk=0.9 (high), multiplier will be ~1.2 giving ~11% on 9% avg
            risk_multiplier = 0.8 + (0.4 * manager.strategy_risk)  # Maps 0.1->0.84, 0.9->1.16
            manager.realized_yield = average_net_apy_realdata * risk_multiplier
            
            # Calculate absolute returns for this manager in this epoch
            manager.absolute_returns = manager.allocated_capital * manager.realized_yield * (self.epoch_length_hours / (24 * 365))
            capital_at_end += manager.allocated_capital + manager.absolute_returns
            
            # Calculate loss buffer as minimum distance from liquidation
            min_buffer = min(health_factors) - 1 + self.strategy_liq_buffer
            manager.loss_buffer = min_buffer
            
            # Check for penalising condition: manager must at least make
            # minimums above bid floor else they get penalised
            manager.to_penalise = self._check_yield_slashing(manager, bid_floor)

            # Calculate raw score
            raw_score, str_score = scoring.calculate_score(manager, self.dao_params)
            manager.score = raw_score
            manager.str_score = str_score
            
        # Second pass: normalize scores and update reputations
        scoring.normalize_scores(list(self.managers.values()))
        
        # Third pass: update reputations with normalized scores
        current_time = epoch_data.index[-1]
        for manager in self.managers.values():
            manager.score_history.add(current_time, manager.score)
            
            # Update reputation using normalized score
            time_delta = manager.score_history.get_last_time_delta()
            if time_delta == 0:
                time_delta = self.epoch_length_hours
            
            manager.reputation = scoring.calculate_ema_reputation(
                manager=manager,
                time_delta=time_delta,
                averaging_window=self.dao_params.ema_window_reputation
            )
        
        # cache previous capital
        self.previous_capital = self.capital
        self.capital = capital_at_end
        
        # Finally allocate capital using normalized scores
        allocations = scoring.allocate_capital(list(self.managers.values()), self.capital)
        for manager_id, alloc in allocations.items():
            
            mngr = self.managers[manager_id]
            starting_allocation = mngr.allocated_capital
            mngr.allocated_capital = alloc
            
            print("----")
            print(f"manager {mngr.manager_id} stats for epoch")
            print(f"manager was allocated {starting_allocation} at the start")
            print(f"manager promised {mngr.promised_yield} and realised {mngr.realized_yield}")
            print(f"manager's absolute profits in epoch: {mngr.absolute_returns}")
            print(f"manager scored {mngr.score} and was allocated {mngr.allocated_capital}")
            print(manager.str_score)
            print(f"new reputation score: {mngr.reputation}")
        
        # Record epoch
        epoch_record = {
            'epoch_id': epoch_id,
            'timestamp': epoch_data.index[0],
            'total_capital': self.capital,
            'ema_risk_free': ema_risk_free,
            'bid_floor': bid_floor,
            'allocations': allocations.copy(),
            'scores': {m_id: m.score for m_id, m in self.managers.items()},
            'promised_yields': {m_id: m.promised_yield for m_id, m in self.managers.items()},
            'realized_yields': {m_id: m.realized_yield for m_id, m in self.managers.items()},
            'bond_amount': {m_id: m.bond_amount for m_id, m in self.managers.items()},
            'absolute_returns': {m_id: m.absolute_returns for m_id, m in self.managers.items()},
            'reputation': {m_id: m.reputation for m_id, m in self.managers.items()},
        }
        self.history.append(epoch_record)
        
        # Prepare for next epoch
        self._handle_slashing_rotation()
        
        print("-------")
        print("epoch total cap", self.capital)
        print("-------\n")
        
        return epoch_record
    
    def run_simulation(self) -> pd.DataFrame:
        """Run full simulation over all historical data."""
        results = []
        
        for epoch_id in range(len(self.resampled_data)):
            epoch_result = self.simulate_epoch(epoch_id)
            results.append(epoch_result)
            
        return pd.DataFrame(results).set_index('timestamp')

    def _check_yield_slashing(self, manager: CapitalManager, bid_floor: float) -> bool:
        """Determine if a manager should be slashed."""
        
        # Slash if yield below floor or bond below minimum
        yield_slash = manager.realized_yield < bid_floor
        return yield_slash
    
    def _check_insufficient_bond(self, manager: CapitalManager) -> bool:
        """Determine if a manager has less bond than minimums."""

        if manager.bond_amount < self.dao_params['min_bond']:
            return True
        
        return False

    def _handle_slashing_rotation(self) -> None:
        """Handle slashing consequences and manager rotation."""
        
        for manager in list(self.managers.values()):  # Create a copy of values
            
            # Apply bond slashing if needed
            if manager.to_penalise:
                manager.bond_amount *= (1 - self.dao_params.slash_percent)
                
            # Set reputation to zero if bond is less than minimum
            if manager.bond_amount < self.dao_params.min_bond:
                manager.reputation = 0

    def _simulate_position_health(
            self, manager: CapitalManager, periods: int, strategy_liq_buffer: float
        ) -> List[float]:
        
        # Base health based on strategy risk (riskier strategies run tighter buffers)
        # Scale buffer based on risk - high risk = closer to min buffer, low risk = more buffer
        # Clamp additional buffer between 0.01 and 0.05 to keep total between 1.07-1.11
        additional_buffer = 0.05 * (1 - manager.strategy_risk)  # 0.05 when risk=0, 0 when risk=1
        base_health = 1 + strategy_liq_buffer + additional_buffer
        
        # Simulate health factors with random walk
        health_factors = []
        current_health = base_health
        for _ in range(periods):
            
            # Add noise based on strategy risk
            # Tiny perturbations scaled by strategy risk, using much smaller std dev
            movement = np.random.normal(0, 0.00001 * manager.strategy_risk) 
            if current_health + movement > 1 + strategy_liq_buffer:
                current_health += movement
            else:
                current_health = 1+strategy_liq_buffer +0.001
            
            health_factors.append(current_health)
        
        return health_factors
