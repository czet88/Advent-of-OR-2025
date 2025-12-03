import pandas as pd
import numpy as np
import os
import yfinance as yf


def extract_market_data(tickers: dict, start_date: str, end_date: str) -> pd.DataFrame:
    # Download historical market data for the given ticker
    my_data = yf.download(list(tickers.keys()), start=start_date, end=end_date, auto_adjust=True)['Close'].reset_index()
    my_data.columns = ['Date'] + list(tickers.values())
    my_data['Date'] = pd.to_datetime(my_data['Date'])
    my_data['Quarter'] = my_data['Date'].dt.to_period('Q')
    my_data.sort_values(by=['Quarter', 'Date'], inplace=True)
    
    # Market summary: last closing price of each quarter
    df_market_summary = my_data.groupby(['Quarter'])[list(tickers.values())].last().reset_index()
    df_market_summary.set_index('Quarter', inplace=True)
    
    # Market summary percentage changes
    summary = round(df_market_summary.pct_change(),2)
    summary.dropna(inplace=True)

    # Select profitability segments based on quantiles
    means = summary.quantile([0.4, 0.5, 0.75, 0.9])
    means.index = ['Prime', 'Standard', 'Substandard', 'Subprime']

    covariance = summary.cov()
    
    return means, covariance

class DataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_scenario(self, config, market_tickers, means, covariance, output_dir="."):
        """
        Generates the 3 CSV files based on a configuration dictionary.
        """
        print(f"Generating scenario: {config['name']}...")
    
        # -- Generate Assets Data --
        assets_data = {
            'asset': list(market_tickers.values()),
            # Crisis = lower ability to change positions (liquidity crunch) -> tighter bounds
            'max_exposure_decrease': np.random.uniform(config['min_liquidity'], config['max_liquidity'], len(market_tickers)),
            'max_exposure_increase': np.random.uniform(config['min_liquidity'], config['max_liquidity'], len(market_tickers)),
        }
        assets_df = pd.DataFrame(assets_data)

        # --- 2. Generate Correlation Data ---
        corr_df = covariance.copy()
        corr_df.insert(0, 'asset', list(market_tickers.values()))
        # --- 3. Generate Segments Data ---
        segments_list = []
        for asset in means.columns:
            for segment in means.index:
                # Different segments have different risk profiles (Prime vs Subprime logic)
                # We simulate this by randomizing the "riskiness" of the segment
                risk_factor = np.random.rand() # 0 = Safe, 1 = Risky
                
                # Risky segments have higher yield (avg profit) but higher risk weight
                base_profit = means.loc[segment,asset]
                economic_factor = config['base_profitability']
                profit_spread = config['profitability_spread']
                
                # In a "Wealth" scenario, base_profit is high. In "Crisis", it might be negative.
                avg_profit = (1+economic_factor)* base_profit + (risk_factor * profit_spread) + np.random.normal(0, 0.005)
                
                risk_weight = config['base_risk_weight'] + (risk_factor * config['risk_weight_spread'])
                
                segments_list.append({
                    'asset': asset,
                    'segment_id': f"{asset}_{segment}",
                    'exposure': int(np.random.uniform(1000, 100000)),
                    'average_profitability': round(avg_profit, 4),
                    'risk_weight': round(risk_weight, 2),
                    # Transaction costs often higher for riskier/illiquid assets
                    'rel_sell_cost': round(np.random.uniform(0.01, 0.05) + (risk_factor * 0.02), 3),
                    'rel_origination_cost': round(np.random.uniform(0.01, 0.05) + (risk_factor * 0.02), 3)
                })
        
        segments_df = pd.DataFrame(segments_list)

        # --- Save Files ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        assets_df.to_csv(f"{output_dir}/assets.csv", index=False)
        corr_df.to_csv(f"{output_dir}/covariance.csv", index=False)
        segments_df.to_csv(f"{output_dir}/segments.csv", index=False)
        print(f"Files saved to {output_dir}/")


# --- SCENARIO DEFINITIONS ---
"""
HOW TO CREATE DIFFERENT SCENARIOS THROUGH PARAMETERIZATION:

Liquidity (max_exposure_decrease/increase): Controls how much can be traded.
CRISIS: Low (e.g., 0.01-0.05) | WEALTH: High (e.g., 0.3-0.5)

Volatility (stdev_profitability): Higher values increase the penalty in the Variance objective.
CRISIS: High (e.g., 0.8+) | WEALTH: Low (e.g., 0.05-0.15)

Correlation: Higher correlation breaks diversification.
CRISIS: High (e.g., 0.9) | WEALTH: Low (e.g., 0.1)

Profitability: Controls the potential return.
CRISIS: Low/Negative (e.g., -0.02) | WEALTH: High (e.g., 0.08)

Risk Weight: Represents the regulatory capital cost. Higher = less risk budget.
CRISIS: High (e.g., 0.5+) | WEALTH: Low (e.g., 0.1)
"""

SCENARIOS = {
    # 1. Large Scale: Big instance for performance testing
    "large_scale": {
        "name": "Large Scale Performance Test",
        "min_liquidity": 0.05, "max_liquidity": 0.20, # Normal liquidity
        "min_volatility": 0.10, "max_volatility": 0.50, # Normal volatility
        "base_correlation": 0.3, "correlation_noise": 0.5, # Normal correlation
        "base_profitability": 0.03, "profitability_spread": 0.05,
        "base_risk_weight": 0.2, "risk_weight_spread": 0.8
    },

    # 2. Stress Test / Crisis: High Correlation, High Volatility, Liquidity Crunch
    "stress_test": {
        "name": "2008 Financial Crisis Mode",
        "min_liquidity": 0.01, "max_liquidity": 0.05, # Can't move money easily!
        "min_volatility": 0.80, "max_volatility": 1.50, # Extreme volatility
        "base_correlation": 0.90, "correlation_noise": 0.1, # Everything crashes together (high rho)
        "base_profitability": -0.02, "profitability_spread": 0.04, # Low/Negative yields
        "base_risk_weight": 0.5, "risk_weight_spread": 1.0 # High risk assets
    },

    # 3. Wealth / Boom: High Yield, Low Volatility, Low Correlation
    "wealth_boom": {
        "name": "Golden Age of Growth",
        "min_liquidity": 0.20, "max_liquidity": 0.50, # High liquidity
        "min_volatility": 0.05, "max_volatility": 0.15, # Stable markets
        "base_correlation": 0.10, "correlation_noise": 0.3, # Diversification works perfectly
        "base_profitability": 0.08, "profitability_spread": 0.10, # Everyone makes money
        "base_risk_weight": 0.1, "risk_weight_spread": 0.4 # Low risk env
    },

    # 4. Instability: Mixed signals, random volatility, unpredictable
    "instability": {
        "name": "Market Instability",
        "min_liquidity": 0.05, "max_liquidity": 0.30,
        "min_volatility": 0.10, "max_volatility": 2.00, # Massive difference between stable and unstable assets
        "base_correlation": 0.0, "correlation_noise": 1.0, # Pure chaos in correlation
        "base_profitability": 0.00, "profitability_spread": 0.20, # Winners and big losers
        "base_risk_weight": 0.2, "risk_weight_spread": 1.5
    }
}

if __name__ == "__main__":
    generator = DataGenerator()
    base_folder = os.path.join("instance_generation", "data")
    os.makedirs(base_folder, exist_ok=True)
    
    # Fetch Market Information
    tickers_df = pd.read_csv(os.path.join("instance_generation", "market_tickers.csv"), index_col=0)
    market_tickers = tickers_df.to_dict(orient="dict")["Loan"]
    means, covariance = extract_market_data(market_tickers, "2018-12-01", "2025-11-30")

    # Generate the requested files
    # You can change the key to generate different folders
    generator.generate_scenario(SCENARIOS["stress_test"], market_tickers, means, covariance, output_dir=os.path.join(base_folder, "stress_test_data"))
    generator.generate_scenario(SCENARIOS["large_scale"], market_tickers, means, covariance, output_dir=os.path.join(base_folder, "large_scale_data"))
    generator.generate_scenario(SCENARIOS["wealth_boom"], market_tickers, means, covariance, output_dir=os.path.join(base_folder, "wealth_boom_data"))
    generator.generate_scenario(SCENARIOS["instability"], market_tickers, means, covariance, output_dir=os.path.join(base_folder, "instability_data"))