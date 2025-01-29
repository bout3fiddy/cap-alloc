"""
Main execution script for capital allocation simulation.
"""

import pandas as pd
import numpy as np

from src.models import CapitalManager, DaoParams
from src.simulation import AllocationSimulator

def main():
    
    # Define simulation constants
    # Set DAO governance parameters   
    dao_params = DaoParams(
        alpha=0.6,
        beta=0.3,
        premium=0.02,
        min_bond=50000,
        penalty=0.01,
        epoch_length_hours=24*14,
        auction_length=-1,
        ema_window_risk_free=24*7,
        ema_window_reputation=30*24,
        slash_percent=0.01
    )
    
    # ----------------------------------------------------------------------
    # Load and prepare historical data
    
    data = pd.read_csv(
        'data/into_the_block_data_2024.csv',
        parse_dates=['DateTime'],
        index_col='DateTime'
    ).drop(columns=["Borrow APY", "Supply APY"]).rename(
        columns={
            'Net APY': 'net_apy',
            'Staking APY': 'staking_apy'
        }
    )
    
    # assuming a 10x leverage on the data, we simply take 
    # the risk free rate (stETH base rate) as:
    data['staking_apy'] = data['staking_apy'] / 10
    data.rename(columns={'staking_apy': 'risk_free_rate'}, inplace=True)
    
    # the risk free rate goes through an exponential moving average (7-D)
    data["ema_riskfree"] = data["risk_free_rate"].ewm(
        span=7*24,  # 7 days * 24 hours
        adjust=False  # Use traditional EMA calculation
    ).mean()
    
    data.drop(columns=['risk_free_rate'], inplace=True)
    
    # ---------------------------------------------------------------------- 
    # Initialize capital managers and run simulation
    
    # 5 capital managers with different strategies
    managers = [
        CapitalManager(
            manager_id=f"CM_{i+1}",
            promised_yield=0,  # initialise it as zero and promise later after revealing bid floor
            strategy_risk=np.random.uniform(0.1, 0.9),
            bond_amount=100000,
            realized_yield=0,
            allocated_capital=0,
            absolute_returns=0,
            score=0,
            reputation=1,
            score_history=None,  # Will be initialized to empty ScoreHistory
            to_penalise=False,
        ) for i in range(5)
    ]

    # ---------------------------------------------------------------------- 
    # Simulate!
    
    simulator = AllocationSimulator(
        data=data,
        initial_managers=managers,
        dao_params=dao_params,
        initial_capital=10_000_000,
        strategy_liq_buffer=0.06,
    )
    
    print("Starting simulation...")
    results = simulator.run_simulation()
    
    # Analyze and display results
    print("Simulation complete. Results summary:")
    print(f"Total epochs simulated: {len(results)}")
    print(f"Final total capital: {results.iloc[-1]['total_capital']:,.2f}")
    
    # plot_results(results, managers)



if __name__ == "__main__":
    main()