import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict, Tuple

# Add current dir and builder dir to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'tkan_dataset_builder'))

from tkan_dataset_builder.data_collector import TkanDataCollector
from tkan_dataset_builder.graph_feature_engineer import GraphFeatureEngineer
from tkan_dataset_builder.advanced_tree_models_optimized import StochasticVol2DTrinomialTree
from tkan_dataset_builder.graph_builder import label_spot_swings_adaptive
from db_config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_intelligence_dashboard(date_str: str, output_path: str = None):
    """
    Generates an interactive HTML dashboard for Alpha Gamma Market Intelligence.
    """
    logging.info(f"Generating Market Intelligence for {date_str}...")
    
    # 1. Initialize Components
    tree_model = StochasticVol2DTrinomialTree()
    collector  = TkanDataCollector(DB_CONFIG)
    engineer   = GraphFeatureEngineer(tree_model, r=0.065)
    
    # 2. Fetch Data
    spot_df, options_df, expiry = collector.fetch_day_data(date_str)
    if spot_df is None or options_df is None or spot_df.empty or options_df.empty:
        logging.error(f"No data found for {date_str}")
        return
    
    expiry_dt = datetime.strptime(expiry, '%Y-%m-%d')
    
    # 3. Label Spot Swings (Adaptive)
    # Match building logic for consistency
    spot_df = spot_df.copy()
    raw_labels = label_spot_swings_adaptive(spot_df, threshold=50) # Use default 50pt
    spot_df['label'] = raw_labels
    
    # 4. Compute Graph State Series
    logging.info("Computing Graph States (This may take a minute)...")
    options_ts = sorted(options_df['ts'].unique())
    mm_inventory = {}
    prev_options = None
    prev_spot = spot_df.iloc[0]['close']
    
    all_times = []
    strike_matrix = [] # [Time, Strike]
    gex_matrix = []
    asy_matrix = []
    pen_matrix = []
    
    for ts in tqdm(options_ts, desc="Extracting Features"):
        if ts not in spot_df.index: continue
        
        ts_options = options_df[options_df['ts'] == ts].copy()
        ts_spot = spot_df.loc[ts, 'close']
        
        graph_dict = engineer.compute_graph_state(
            spot=ts_spot,
            spot_prev=prev_spot,
            options_df=ts_options,
            options_prev_df=prev_options,
            expiry_dt=expiry_dt,
            mm_inventory=mm_inventory
        )
        
        all_times.append(ts)
        strike_matrix.append(graph_dict['node_strikes'])
        gex_matrix.append(graph_dict['nodes'][:, 0]) # Net GEX
        asy_matrix.append(graph_dict['nodes'][:, 1]) # Asymmetry
        
        # Pull Penetration Energy from Global Agent features (index 2)
        pen_matrix.append(graph_dict['global_price'][:, 2])
        
        prev_options = ts_options
        prev_spot = ts_spot

    # Convert to NumPy for easier indexing
    times_arr = np.array(all_times)
    strikes_arr = np.array(strike_matrix[0]) # Assuming static range ±15 strikes
    gex_arr = np.array(gex_matrix)
    pen_arr = np.array(pen_matrix)
    
    # 5. Build Dashboard (Plotly)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.4, 0.2],
        subplot_titles=(
            "NIFTY Spot & Adaptive Swings",
            "Gamma Force Field (Net GEX per Strike)",
            "Penetration Energy (Price Acceleration / GEX Resistance)"
        )
    )

    # Panel 1: Spot Price
    fig.add_trace(
        go.Scatter(x=spot_df.index, y=spot_df['close'], name="NIFTY Spot", line=dict(color='black', width=2)),
        row=1, col=1
    )
    
    # Swing Markers
    bullish = spot_df[spot_df['label'] == 1]
    bearish = spot_df[spot_df['label'] == -1]
    
    fig.add_trace(
        go.Scatter(x=bullish.index, y=bullish['close'], mode='markers', name='Bullish Swing',
                   marker=dict(symbol='triangle-up', size=10, color='forestgreen')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=bearish.index, y=bearish['close'], mode='markers', name='Bearish Swing',
                   marker=dict(symbol='triangle-down', size=10, color='crimson')),
        row=1, col=1
    )

    # Panel 2: Net GEX Heatmap
    # Y-axis: Strikes, X-Axis: Time, Z-Axis: GEX
    fig.add_trace(
        go.Heatmap(
            z=gex_arr.T,
            x=times_arr,
            y=strikes_arr,
            colorscale='RdBu',
            reversescale=True,
            zmid=0,
            name="Net GEX",
            colorbar=dict(title="GEX Magnitude", thickness=15, len=0.3, y=0.5)
        ),
        row=2, col=1
    )

    # Panel 3: Penetration Energy
    fig.add_trace(
        go.Heatmap(
            z=pen_arr.T,
            x=times_arr,
            y=strikes_arr,
            colorscale='Viridis',
            name="Penetration Energy",
            colorbar=dict(title="Penetration", thickness=15, len=0.2, y=0.1)
        ),
        row=3, col=1
    )

    # Layout Adjustments
    fig.update_layout(
        title=dict(text=f"Alpha Gamma Market Intelligence - {date_str}", x=0.5, font=dict(size=24)),
        height=1000,
        showlegend=True,
        template='plotly_white',
    )
    
    fig.update_yaxes(title_text="NIFTY Level", row=1, col=1)
    fig.update_yaxes(title_text="Strike Price", row=2, col=1)
    fig.update_yaxes(title_text="Strike Price", row=3, col=1)

    if not output_path:
        output_path = f"alpha_gamma_intelligence_{date_str}.html"
    
    fig.write_html(output_path)
    logging.info(f"Dashboard saved to: {output_path}")

from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpha Gamma Market Intelligence Visualizer")
    parser.add_argument("--date", type=str, default="2024-01-30", help="Trade date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    create_intelligence_dashboard(args.date)
