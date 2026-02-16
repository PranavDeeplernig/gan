import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from gex_market_structure import GEXCalculator
from advanced_tree_models_optimized import StochasticVol2DTrinomialTree

class GraphFeatureEngineer:
    def __init__(self, model: StochasticVol2DTrinomialTree, r: float, n_steps: int = 30):
        self.gex_calc = GEXCalculator(model, r, n_steps=n_steps)
        self.strike_step = 50
        self.volume_history = []  # To store total volume of past 5 candles

    def compute_graph_state(self, spot: float, spot_prev: float, options_df: pd.DataFrame, 
                           options_prev_df: pd.DataFrame, expiry_dt: datetime, 
                           mm_inventory: Dict[Tuple[int, str], float]) -> Dict:
        """
        Build the graph representation for a single timestamp.
        """
        ts_dt = options_df['ts'].iloc[0]
        T = max((expiry_dt - ts_dt).total_seconds() / (365.25 * 24 * 3600), 1/365)
        
        # Calculate current total volume
        current_total_vol = options_df['volume'].sum() if 'volume' in options_df.columns else 0.0
        self.volume_history.append(current_total_vol)
        if len(self.volume_history) > 5:
            self.volume_history.pop(0)
            
        # Use average of past 5 candles (or what's available)
        avg_vol_5_candles = np.mean(self.volume_history) if self.volume_history else 1.0
        if avg_vol_5_candles == 0: avg_vol_5_candles = 1.0

        # ... (rest of the logic)
        atm = int(round(spot / 50) * 50)
        strikes_range = np.array([atm + i * 50 for i in range(-15, 16)])
        
        opt_map = options_df.set_index(['strike', 'option_type'])
        opt_prev_map = options_prev_df.set_index(['strike', 'option_type']) if options_prev_df is not None and not options_prev_df.empty else None
        
        node_features = []
        node_strikes = []
        
        for strike in strikes_range:
            node_strikes.append(strike)
            
            def get_data(s, t, df_map):
                if df_map is not None and (s, t) in df_map.index:
                    row = df_map.loc[(s, t)]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]
                    return (float(row['mkt_close']), float(row['iv']), 
                            float(row['oi']), float(row.get('volume', 0)))
                return (0.0, 0.0, 0.0, 0.0)

            ce_px, ce_iv, ce_oi, ce_vol = get_data(strike, 'CE', opt_map)
            pe_px, pe_iv, pe_oi, pe_vol = get_data(strike, 'PE', opt_map)
            
            ce_prev_px, ce_prev_iv, ce_prev_oi, ce_prev_vol = get_data(strike, 'CE', opt_prev_map)
            pe_prev_px, pe_prev_iv, pe_prev_oi, pe_prev_vol = get_data(strike, 'PE', opt_prev_map)
            
            ce_oi_delta = ce_oi - ce_prev_oi
            pe_oi_delta = pe_oi - pe_prev_oi
            
            strike_arr = np.array([strike, strike])
            type_arr = np.array(['CE', 'PE'])
            px_arr = np.array([ce_px, pe_px])
            iv_arr = np.array([ce_iv, pe_iv])
            oid_arr = np.array([ce_oi_delta, pe_oi_delta])
            oi_arr = np.array([ce_oi, pe_oi])
            vol_arr = np.array([ce_vol, pe_vol])
            # Determine if we should use inventory or absolute OI
            if mm_inventory:
                inv_arr = np.array([mm_inventory.get((strike, 'CE'), 0.0), 
                                   mm_inventory.get((strike, 'PE'), 0.0)])
            else:
                inv_arr = None
            
            gex_df = self.gex_calc.compute_strike_gex(
                spot=spot, T=T, t_now=ts_dt,
                strikes=strike_arr, option_types=type_arr,
                mkt_prices=px_arr, ivs=iv_arr,
                oi_deltas=oid_arr, absolute_oi=oi_arr,
                mm_inventory=inv_arr, volumes=vol_arr
            )
            
            net_gex = gex_df['heatmap_gex'].sum()
            # Normalize by 5-candle average volume
            n_gex = net_gex / (avg_vol_5_candles + 1e-9)
            
            ce_gex_val = float(gex_df[gex_df['option_type'] == 'CE']['heatmap_gex'].values[0])
            pe_gex_val = float(gex_df[gex_df['option_type'] == 'PE']['heatmap_gex'].values[0])
            g_asy = (abs(ce_gex_val) - abs(pe_gex_val)) / (abs(ce_gex_val) + abs(pe_gex_val) + 1e-9)
            
            # Normalize charm and vanna by spot to make them comparable
            charm = gex_df['charm'].sum() / spot
            vanna = gex_df['vanna'].sum() / spot
            
            node_features.append([n_gex, g_asy, charm, vanna])

        node_features = np.array(node_features, dtype=np.float32)
        
        # 2. Edges
        edges = []
        # Spatial Edges (Immediate neighbors)
        for i in range(len(node_strikes) - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
            
        # Gamma Resonance Edges
        # Connect nodes with high GEX magnitude
        gex_mags = np.abs(node_features[:, 0])
        if len(gex_mags) > 0:
            threshold = np.percentile(gex_mags, 90)
            high_gex_nodes = np.where(gex_mags > threshold)[0]
            for i in range(len(high_gex_nodes)):
                for j in range(i + 1, len(high_gex_nodes)):
                    u, v = high_gex_nodes[i], high_gex_nodes[j]
                    edges.append([u, v])
                    edges.append([v, u])
                    
        # 3. Price Agent Features (per node relative to spot)
        # 1. dist_to_node
        # 2. velocity (spot ROC)
        # 3. penetration_energy (ROC / abs(GEX))
        
        roc = (spot - spot_prev) / (spot_prev + 1e-9)
        price_agent_features = []
        for i, strike in enumerate(node_strikes):
            dist = (strike - spot) / spot # normalized distance
            gex_val = abs(node_features[i, 0])
            pen_energy = roc / (gex_val + 1e-9)
            price_agent_features.append([dist, roc, pen_energy])
            
        return {
            'nodes': node_features,
            'node_strikes': node_strikes,
            'edges': np.array(edges),
            'global_price': np.array(price_agent_features, dtype=np.float32)
        }
