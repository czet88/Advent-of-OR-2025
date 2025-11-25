import pandas as pd
import numpy as np
import os

class DataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_correlation_matrix(self, n_assets, base_correlation=0.0, noise=0.1):
        """
        Generates a valid correlation matrix.
        base_correlation: The average correlation between valid distinct assets.
                          Higher = Crisis (everything moves together).
                          Lower = Diversification (uncorrelated assets).
        """
        # Start with a random matrix
        A = np.random.normal(size=(n_assets, n_assets))
        
        # Create a covariance matrix (A * A.T) to ensure it's positive semi-definite
        cov = np.dot(A, A.T)
        
        # Normalize to get correlation (diagonals = 1)
        d = np.diag(cov)
        corr = cov / np.sqrt(np.outer(d, d))
        
        # Apply the base_correlation "bias" to the off-diagonal elements
        # We blend the random valid correlation matrix with a fixed matrix of value 'base_correlation'
        # This is a heuristic to steer the matrix properties without breaking mathematical validity
        mask = np.ones((n_assets, n_assets)) - np.eye(n_assets)
        corr = corr * (1 - mask) + (corr * noise + base_correlation * (1 - noise)) * mask
        
        # Ensure perfect 1s on diagonal and symmetry (floating point correction)
        np.fill_diagonal(corr, 1.0)
        corr = (corr + corr.T) / 2
        
        # Clip to [-1, 1] just in case
        corr = np.clip(corr, -1.0, 1.0)
        
        return corr

    def generate_scenario(self, config, output_dir="."):
        """
        Generates the 3 CSV files based on a configuration dictionary.
        """
        print(f"Generating scenario: {config['name']}...")
        
        n_assets = config['n_assets']
        n_segments_per_asset = config['n_segments_per_asset']
        
        # --- 1. Generate Assets Data ---
        asset_names = [f"Asset_{i:02d}" for i in range(n_assets)]
        
        assets_data = {
            'asset': asset_names,
            # Crisis = lower ability to change positions (liquidity crunch) -> tighter bounds
            'max_exposure_decrease': np.random.uniform(config['min_liquidity'], config['max_liquidity'], n_assets),
            'max_exposure_increase': np.random.uniform(config['min_liquidity'], config['max_liquidity'], n_assets),
            # Crisis = High Volatility
            'stdev_profitability': np.random.uniform(config['min_volatility'], config['max_volatility'], n_assets)
        }
        assets_df = pd.DataFrame(assets_data)

        # --- 2. Generate Correlation Data ---
        corr_matrix = self.generate_correlation_matrix(
            n_assets, 
            base_correlation=config['base_correlation'], 
            noise=config['correlation_noise']
        )
        corr_df = pd.DataFrame(corr_matrix, columns=asset_names)
        corr_df.insert(0, 'asset', asset_names)

        # --- 3. Generate Segments Data ---
        segments_list = []
        for asset in asset_names:
            for i in range(n_segments_per_asset):
                # Different segments have different risk profiles (Prime vs Subprime logic)
                # We simulate this by randomizing the "riskiness" of the segment
                risk_factor = np.random.rand() # 0 = Safe, 1 = Risky
                
                # Risky segments have higher yield (avg profit) but higher risk weight
                base_profit = config['base_profitability']
                profit_spread = config['profitability_spread']
                
                # In a "Wealth" scenario, base_profit is high. In "Crisis", it might be negative.
                avg_profit = base_profit + (risk_factor * profit_spread) + np.random.normal(0, 0.005)
                
                risk_weight = config['base_risk_weight'] + (risk_factor * config['risk_weight_spread'])
                
                segments_list.append({
                    'asset': asset,
                    'segment_id': f"{asset}_Seg_{i:02d}",
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
        corr_df.to_csv(f"{output_dir}/correlation.csv", index=False)
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
        "n_assets": 50,
        "n_segments_per_asset": 20, # Total 1000 variables
        "min_liquidity": 0.05, "max_liquidity": 0.20, # Normal liquidity
        "min_volatility": 0.10, "max_volatility": 0.50, # Normal volatility
        "base_correlation": 0.3, "correlation_noise": 0.5, # Normal correlation
        "base_profitability": 0.03, "profitability_spread": 0.05,
        "base_risk_weight": 0.2, "risk_weight_spread": 0.8
    },

    # 2. Stress Test / Crisis: High Correlation, High Volatility, Liquidity Crunch
    "stress_test": {
        "name": "2008 Financial Crisis Mode",
        "n_assets": 10,
        "n_segments_per_asset": 5,
        "min_liquidity": 0.01, "max_liquidity": 0.05, # Can't move money easily!
        "min_volatility": 0.80, "max_volatility": 1.50, # Extreme volatility
        "base_correlation": 0.90, "correlation_noise": 0.1, # Everything crashes together (high rho)
        "base_profitability": -0.02, "profitability_spread": 0.04, # Low/Negative yields
        "base_risk_weight": 0.5, "risk_weight_spread": 1.0 # High risk assets
    },

    # 3. Wealth / Boom: High Yield, Low Volatility, Low Correlation
    "wealth_boom": {
        "name": "Golden Age of Growth",
        "n_assets": 10,
        "n_segments_per_asset": 5,
        "min_liquidity": 0.20, "max_liquidity": 0.50, # High liquidity
        "min_volatility": 0.05, "max_volatility": 0.15, # Stable markets
        "base_correlation": 0.10, "correlation_noise": 0.3, # Diversification works perfectly
        "base_profitability": 0.08, "profitability_spread": 0.10, # Everyone makes money
        "base_risk_weight": 0.1, "risk_weight_spread": 0.4 # Low risk env
    },

    # 4. Instability: Mixed signals, random volatility, unpredictable
    "instability": {
        "name": "Market Instability",
        "n_assets": 15,
        "n_segments_per_asset": 8,
        "min_liquidity": 0.05, "max_liquidity": 0.30,
        "min_volatility": 0.10, "max_volatility": 2.00, # Massive difference between stable and unstable assets
        "base_correlation": 0.0, "correlation_noise": 1.0, # Pure chaos in correlation
        "base_profitability": 0.00, "profitability_spread": 0.20, # Winners and big losers
        "base_risk_weight": 0.2, "risk_weight_spread": 1.5
    }
}

if __name__ == "__main__":
    generator = DataGenerator()
    
    # Generate the requested files
    # You can change the key to generate different folders
    generator.generate_scenario(SCENARIOS["stress_test"], output_dir="stress_test_data")
    generator.generate_scenario(SCENARIOS["large_scale"], output_dir="large_scale_data")
    generator.generate_scenario(SCENARIOS["wealth_boom"], output_dir="wealth_boom_data")
    generator.generate_scenario(SCENARIOS["instability"], output_dir="instability_data")
