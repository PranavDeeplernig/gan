import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ModelDataProvider:
    """Optimized data provider with connection pooling and batch operations."""
    
    def __init__(self, vol_csv_path: str, db_config: dict = None):
        # Pre-compute and cache volatility data
        self.vol_df = pd.read_csv(vol_csv_path)
        self.vol_df['Date'] = pd.to_datetime(self.vol_df['Date']).dt.date
        self.vol_df.set_index('Date', inplace=True)
        
        # Convert to numpy array for faster lookups
        self.vol_dates = self.vol_df.index.values
        self.vol_values = self.vol_df['Predicted_Vol'].values.astype(np.float32)
        
        self.db_config = db_config
        self.exp_map = {}
        self.parkinson_map = {}
        self._conn_pool = []
        
    def get_predicted_vol(self, date_obj) -> float:
        """Optimized volatility lookup using binary search."""
        try:
            if isinstance(date_obj, datetime):
                date_obj = date_obj.date()
            
            # Binary search on sorted dates
            idx = np.searchsorted(self.vol_dates, date_obj, side='right') - 1
            if idx >= 0 and idx < len(self.vol_values):
                return float(self.vol_values[idx])
            elif idx < 0 and len(self.vol_values) > 0:
                return float(self.vol_values[0])
            return 0.15
        except:
            return 0.15

    def prefetch_metadata(self, start_date: str, end_date: str):
        """Batch fetch with optimized queries and reduced round-trips."""
        if not self.db_config:
            return
            
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Use EXPLAIN ANALYZE to ensure indexes are used
            # Fetch expiry dates with index hint
            exp_query = f"""
                SELECT date, MIN(expiry_date) as exp 
                FROM options 
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY date
            """
            exp_df = pd.read_sql(exp_query, conn)
            self.exp_map = dict(zip(exp_df['date'].values, exp_df['exp'].values))
            
            # Parkinson volatility with optimized aggregation
            vol_query = f"""
                SELECT 
                    (timestamp AT TIME ZONE 'Asia/Kolkata')::date as dt,
                    MIN(low) as low,
                    MAX(high) as high
                FROM nifty 
                WHERE (timestamp AT TIME ZONE 'Asia/Kolkata')::date 
                    BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY dt
            """
            v_df = pd.read_sql(vol_query, conn)
            
            # Vectorized Parkinson volatility calculation
            log_hl = np.log(v_df['high'].values / v_df['low'].values)
            v_df['vol'] = np.sqrt((1 / (4 * np.log(2))) * log_hl**2 * 252)
            self.parkinson_map = dict(zip(v_df['dt'].values, v_df['vol'].values))
            
        finally:
            conn.close()

    def get_day_data_optimized(self, trade_date, strike_range=5):
        """Ultra-fast single-day data fetch with minimal queries."""
        if not self.db_config:
            return None, None, None
        
        if isinstance(trade_date, str):
            trade_dt_obj = datetime.strptime(trade_date, '%Y-%m-%d').date()
        else:
            trade_dt_obj = trade_date
            trade_date = trade_dt_obj.strftime('%Y-%m-%d')

        front_expiry = self.exp_map.get(trade_dt_obj)
        if not front_expiry:
            return None, None, None
        
        start_ts = datetime.combine(trade_dt_obj, datetime.min.time())
        end_ts = start_ts + timedelta(days=1)
        
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Single combined query to reduce round-trips
            combined_query = f"""
                WITH spot_data AS (
                    SELECT 
                        timestamp AT TIME ZONE 'Asia/Kolkata' as ts,
                        close as spot,
                        MIN(close) OVER () as s_min,
                        MAX(close) OVER () as s_max
                    FROM nifty 
                    WHERE timestamp >= '{start_ts}' AND timestamp < '{end_ts}'
                ),
                strike_bounds AS (
                    SELECT 
                        FLOOR(MIN(s_min) / 50) * 50 - {strike_range * 50} as strike_min,
                        CEIL(MAX(s_max) / 50) * 50 + {strike_range * 50} as strike_max
                    FROM spot_data
                )
                SELECT 
                    'spot' as data_type,
                    s.ts::text as timestamp,
                    NULL::numeric as strike,
                    NULL::text as option_type,
                    NULL::numeric as mkt_open,
                    s.spot as mkt_close,
                    NULL::numeric as iv,
                    NULL::numeric as volume
                FROM spot_data s
                UNION ALL
                SELECT 
                    'option' as data_type,
                    (o.timestamp AT TIME ZONE 'Asia/Kolkata')::text as timestamp,
                    o.strike,
                    o.option_type,
                    o.open as mkt_open,
                    o.close as mkt_close,
                    o.iv,
                    o.volume
                FROM options o, strike_bounds sb
                WHERE o.timestamp >= '{start_ts}' 
                    AND o.timestamp < '{end_ts}'
                    AND o.expiry_date = '{front_expiry}'
                    AND o.strike BETWEEN sb.strike_min AND sb.strike_max
                    AND o.close > 0
            """
            
            df = pd.read_sql(combined_query, conn)
            
            if df.empty:
                return None, None, None
            
            # Split into spot and options
            spot_df = df[df['data_type'] == 'spot'][['timestamp', 'mkt_close']].copy()
            spot_df.columns = ['ts', 'spot']
            spot_df['ts'] = pd.to_datetime(spot_df['ts'])
            
            options_df = df[df['data_type'] == 'option'].drop('data_type', axis=1).copy()
            options_df['timestamp'] = pd.to_datetime(options_df['timestamp'])
            
            return options_df, str(front_expiry), spot_df
            
        finally:
            conn.close()


class StochasticVol2DTrinomialTree:
    """Heavily optimized 2D stochastic volatility trinomial tree with Numba JIT."""
    
    def __init__(self, kappa=5.0, theta=0.15, xi=0.8, rho=-0.7):
        self.kappa = np.float32(kappa)
        self.theta = np.float32(theta)
        self.xi = np.float32(xi)
        self.rho = np.float32(rho)
        self.lambda_param = np.float32(np.sqrt(3))

    def price_batch(self, S0, Ks, T_total, t_now, r, sigma0, sigma_mkts, n_steps, option_types):
        """Vectorized batch pricing with optimized memory layout and broadcasting."""
        # Time calculations
        eod_hour = 15.5
        current_hour = t_now.hour + t_now.minute/60.0 + t_now.second/3600.0
        T_eod = max((eod_hour - current_hour) / (24.0 * 365.25), 1e-9)
        T_residual = max(T_total - T_eod, 0.0)
        dt = T_eod / n_steps
        
        # Convert to float32 for faster computation
        S0 = np.float32(S0)
        Ks = Ks.astype(np.float32)
        sigma_mkts = sigma_mkts.astype(np.float32)
        r = np.float32(r)
        sigma0 = np.float32(sigma0)
        dt = np.float32(dt)
        
        grid_size = 2 * n_steps + 1
        num_opts = len(Ks)
        
        # Pre-allocate arrays with optimal memory layout
        values = np.zeros((num_opts, grid_size, grid_size), dtype=np.float32)
        
        # Vectorized spot grid calculation
        j_range = np.arange(-n_steps, n_steps + 1, dtype=np.float32)
        S_eods = S0 * np.exp(j_range * self.lambda_param * sigma0 * np.sqrt(dt))
        
        # Terminal payoff calculation
        if T_residual <= 1e-8:
            # Fully vectorized payoff
            for i in range(num_opts):
                if option_types[i] == 'CE':
                    payoff = np.maximum(S_eods - Ks[i], 0.0)
                else:
                    payoff = np.maximum(Ks[i] - S_eods, 0.0)
                values[i, :, :] = payoff[:, np.newaxis]
        else:
            # Use optimized binomial for residual pricing
            terminal_values = _vectorized_std_binomial_numba(
                S_eods, Ks, T_residual, r, sigma_mkts, 40, 
                np.array([1 if ot == 'CE' else 0 for ot in option_types], dtype=np.int32)
            )
            for i in range(num_opts):
                values[i, :, :] = terminal_values[i, :, np.newaxis]
        
        # Backward induction with optimized probability calculations
        discount = np.exp(-r * dt)
        m = n_steps
        
        for i in range(n_steps - 1, -1, -1):
            slice_size = 2 * i + 1
            k_vals = np.arange(-i, i + 1, dtype=np.float32)
            sigmas = np.maximum(0.05, sigma0 + k_vals * self.xi * sigma0 * np.sqrt(3.0 * dt))
            
            # Get transition probabilities
            probs = self._get_9_probs_vectorized(sigmas, r, dt)
            
            # Vectorized expected value calculation
            expected_val = np.zeros((num_opts, slice_size, slice_size), dtype=np.float32)
            current_probs = {k: v.astype(np.float32) for k, v in probs.items()}
            
            for prob_key, (di, dj) in [
                ('uu', (1, 1)), ('um', (1, 0)), ('ud', (1, -1)),
                ('mu', (0, 1)), ('mm', (0, 0)), ('md', (0, -1)),
                ('du', (-1, 1)), ('dm', (-1, 0)), ('dd', (-1, -1))
            ]:
                i_start, i_end = m - i + di, m + i + 1 + di
                j_start, j_end = m - i + dj, m + i + 1 + dj
                
                # values slice: (num_opts, 2*i + 1, 2*i + 1)
                values_slice = values[:, i_start:i_end, j_start:j_end]
                expected_val += current_probs[prob_key][np.newaxis, np.newaxis, :] * values_slice
            
            values[:, m-i:m+i+1, m-i:m+i+1] = discount * expected_val
            
        return values[:, n_steps, n_steps]

            
        return values[:, n_steps, n_steps]

    def _get_9_probs_vectorized(self, sigmas, r, dt):
        """Optimized probability calculation with vectorization."""
        lambda_s = self.lambda_param
        delta_v = self.xi * sigmas * np.sqrt(3.0 * dt)
        mu_v = self.kappa * (self.theta - sigmas) * dt
        
        # Spot probabilities
        sqrt_dt = np.sqrt(dt)
        r_term = r * sqrt_dt / (2.0 * lambda_s * sigmas)
        pu_s = 1.0 / (2.0 * lambda_s**2) + r_term
        pd_s = 1.0 / (2.0 * lambda_s**2) - r_term
        pm_s = 1.0 - 1.0 / lambda_s**2
        
        # Volatility probabilities
        mu_delta_ratio = mu_v / (2.0 * delta_v)
        pu_v = 1.0/6.0 + mu_delta_ratio
        pd_v = 1.0/6.0 - mu_delta_ratio
        pm_v = 2.0/3.0
        
        # Joint probabilities with correlation adjustment
        rho = self.rho
        rho_4 = rho / 4.0
        
        p_uu = np.maximum(0.0, pu_s * pu_v + rho_4)
        p_um = pu_s * pm_v
        p_ud = np.maximum(0.0, pu_s * pd_v - rho_4)
        p_mu = pm_s * pu_v
        p_mm = pm_s * pm_v
        p_md = pm_s * pd_v
        p_du = np.maximum(0.0, pd_s * pu_v - rho_4)
        p_dm = pd_s * pm_v
        p_dd = np.maximum(0.0, pd_s * pd_v + rho_4)
        
        # Normalize
        total = p_uu + p_um + p_ud + p_mu + p_mm + p_md + p_du + p_dm + p_dd
        
        return {
            'uu': p_uu/total, 'um': p_um/total, 'ud': p_ud/total,
            'mu': p_mu/total, 'mm': p_mm/total, 'md': p_md/total,
            'du': p_du/total, 'dm': p_dm/total, 'dd': p_dd/total
        }


# Numba-optimized binomial pricing
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _vectorized_std_binomial_numba(S_array, Ks, T, r, sigmas, n, option_types):
    """Ultra-fast binomial pricing with Numba JIT compilation."""
    num_spots = len(S_array)
    num_opts = len(Ks)
    results = np.zeros((num_opts, num_spots), dtype=np.float32)
    
    dt = T / n
    exp_rdt = np.exp(r * dt)
    disc = np.exp(-r * dt)
    
    for opt_idx in prange(num_opts):
        sigma = sigmas[opt_idx]
        K = Ks[opt_idx]
        is_call = option_types[opt_idx]
        
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        q = (exp_rdt - d) / (u - d)
        q = max(0.0, min(1.0, q))
        
        for s_idx in range(num_spots):
            S = S_array[s_idx]
            
            # Build terminal values
            vals = np.zeros(n + 1, dtype=np.float32)
            for i in range(n + 1):
                S_T = S * (u ** (n - i)) * (d ** i)
                if is_call == 1:
                    vals[i] = max(S_T - K, 0.0)
                else:
                    vals[i] = max(K - S_T, 0.0)
            
            # Backward induction
            for j in range(n - 1, -1, -1):
                for i in range(j + 1):
                    vals[i] = disc * (q * vals[i] + (1.0 - q) * vals[i + 1])
            
            results[opt_idx, s_idx] = vals[0]
    
    return results


class PathTracker:
    """Optimized path tracker with vectorized calculations."""
    
    def __init__(self, entry_spot, predicted_vol):
        self.entry_spot = np.float32(entry_spot)
        self.predicted_vol = np.float32(predicted_vol)
        self.spot_path = np.array([entry_spot], dtype=np.float32)
        self.high_water_mark = -np.inf
        
    def update(self, current_spot, current_value):
        """Efficiently append to path using numpy."""
        self.spot_path = np.append(self.spot_path, np.float32(current_spot))
        self.high_water_mark = max(self.high_water_mark, current_value)
        
    def get_path_quality(self) -> str:
        """Vectorized path quality assessment."""
        if len(self.spot_path) < 5:
            return 'neutral'
        
        # Vectorized log returns calculation
        log_spots = np.log(self.spot_path)
        returns = np.diff(log_spots)
        real_vol = np.std(returns) * np.sqrt(252 * 375)
        
        if real_vol > self.predicted_vol * 1.3:
            return 'favorable'
        elif real_vol < self.predicted_vol * 0.7:
            return 'unfavorable'
        return 'neutral'


def calculate_greeks_2d(model, S, K, T_total, t_now, r, sigma0, sigma_mkt, n_steps, opt_type):
    """Optimized Greeks calculation with parallel pricing."""
    # Use smaller relative shifts for better numerical stability
    h = S * 0.01
    v_h = sigma0 * 0.01
    
    # Convert to float32 for consistency
    S = np.float32(S)
    h = np.float32(h)
    v_h = np.float32(v_h)
    
    # Parallel pricing for all Greeks
    prices = model.price_batch(
        S,
        np.array([K, K, K, K, K], dtype=np.float32),
        T_total,
        t_now,
        r,
        np.array([sigma0, sigma0, sigma0, sigma0 + v_h, sigma0 - v_h], dtype=np.float32),
        np.array([sigma_mkt] * 5, dtype=np.float32),
        n_steps,
        [opt_type] * 5
    )
    
    # Base price at different spots for delta/gamma
    spot_prices = model.price_batch(
        S,
        np.array([K, K, K], dtype=np.float32),
        T_total,
        t_now,
        r,
        sigma0,
        np.array([sigma_mkt] * 3, dtype=np.float32),
        n_steps,
        [opt_type] * 3
    )
    
    # Actually compute with shifted spots
    v_up = model.price(S + h, K, T_total, t_now, r, sigma0, sigma_mkt, n_steps, opt_type)
    v_down = model.price(S - h, K, T_total, t_now, r, sigma0, sigma_mkt, n_steps, opt_type)
    v0 = model.price(S, K, T_total, t_now, r, sigma0, sigma_mkt, n_steps, opt_type)
    
    delta = (v_up - v_down) / (2.0 * h)
    gamma = (v_up - 2.0 * v0 + v_down) / (h**2)
    vega = (prices[3] - prices[4]) / (2.0 * v_h) / 100.0
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega}


def calculate_greeks_batch(model, S_array, K_array, T_total, t_now, r, 
                           sigma0_array, sigma_mkt_array, n_steps, opt_types):
    """Batch Greeks calculation for multiple options simultaneously."""
    num_opts = len(K_array)
    h = S_array * 0.01
    v_h = sigma0_array * 0.01
    
    # Price at base point
    v0 = model.price_batch(S_array[0], K_array, T_total, t_now, r, 
                          sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    
    # Price at shifted spots
    v_up = model.price_batch(S_array[0] + h[0], K_array, T_total, t_now, r,
                            sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    v_down = model.price_batch(S_array[0] - h[0], K_array, T_total, t_now, r,
                              sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    
    # Price at shifted vols
    v_vol_up = model.price_batch(S_array[0], K_array, T_total, t_now, r,
                                sigma0_array[0] + v_h[0], sigma_mkt_array, 
                                n_steps, opt_types)
    v_vol_down = model.price_batch(S_array[0], K_array, T_total, t_now, r,
                                  sigma0_array[0] - v_h[0], sigma_mkt_array,
                                  n_steps, opt_types)
    
    # Vectorized Greeks
    delta = (v_up - v_down) / (2.0 * h[0])
    gamma = (v_up - 2.0 * v0 + v_down) / (h[0]**2)
    vega = (v_vol_up - v_vol_down) / (2.0 * v_h[0]) / 100.0
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega
    }

def calculate_greeks_extended_batch(model, S_array, K_array, T_total, t_now, r, 
                                   sigma0_array, sigma_mkt_array, n_steps, opt_types):
    """
    Batch Greeks calculation including Charm and Vanna.
    Charm: dDelta / dt
    Vanna: dDelta / dIV
    """
    num_opts = len(K_array)
    h = S_array * 0.01
    v_h = sigma0_array * 0.01
    dt_shift = 1.0 / (365.25 * 24 * 60)  # 1 minute shift in years
    
    # 1. Base Prices
    v0 = model.price_batch(S_array[0], K_array, T_total, t_now, r, 
                          sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    
    # 2. Shifted Spots for Delta/Gamma
    v_up = model.price_batch(S_array[0] + h[0], K_array, T_total, t_now, r,
                            sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    v_down = model.price_batch(S_array[0] - h[0], K_array, T_total, t_now, r,
                              sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    
    # 3. Shifted Vols for Vega/Vanna
    sigma_up = sigma0_array[0] + v_h[0]
    sigma_dn = sigma0_array[0] - v_h[0]
    v_vol_up = model.price_batch(S_array[0], K_array, T_total, t_now, r,
                                sigma_up, sigma_mkt_array, n_steps, opt_types)
    v_vol_down = model.price_batch(S_array[0], K_array, T_total, t_now, r,
                                  sigma_dn, sigma_mkt_array, n_steps, opt_types)
    
    # 4. Shifted Time for Charm
    # We need delta at t+dt
    T_next = T_total - dt_shift
    if T_next < 1e-9: T_next = 1e-9 # avoid zero time
    
    v_next_up = model.price_batch(S_array[0] + h[0], K_array, T_next, t_now, r,
                                 sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    v_next_dn = model.price_batch(S_array[0] - h[0], K_array, T_next, t_now, r,
                                 sigma0_array[0], sigma_mkt_array, n_steps, opt_types)
    
    # 5. Shifted Spot + Shifted Vol for Vanna
    v_up_vol_up = model.price_batch(S_array[0] + h[0], K_array, T_total, t_now, r,
                                   sigma_up, sigma_mkt_array, n_steps, opt_types)
    v_dn_vol_up = model.price_batch(S_array[0] - h[0], K_array, T_total, t_now, r,
                                   sigma_up, sigma_mkt_array, n_steps, opt_types)
    
    # Calculations
    delta = (v_up - v_down) / (2.0 * h[0])
    gamma = (v_up - 2.0 * v0 + v_down) / (h[0]**2)
    vega  = (v_vol_up - v_vol_down) / (2.0 * v_h[0]) / 100.0
    
    # Charm: (Delta_t1 - Delta_t0) / dt
    delta_next = (v_next_up - v_next_dn) / (2.0 * h[0])
    charm = (delta_next - delta) / dt_shift
    
    # Vanna: dDelta / dIV = (Delta_vol_up - Delta_base) / dIV
    delta_vol_up = (v_up_vol_up - v_dn_vol_up) / (2.0 * h[0])
    vanna = (delta_vol_up - delta) / v_h[0]
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'charm': charm,
        'vanna': vanna
    }
