import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import logging

class TkanDataCollector:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.interval = '3min'

    def get_available_dates(self, start_date: str) -> List[str]:
        """Fetch all available trading dates from the start_date."""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT date FROM options WHERE date >= %s ORDER BY date", (start_date,))
        dates = [d[0].strftime('%Y-%m-%d') for d in cursor.fetchall()]
        conn.close()
        return dates

    def fetch_day_data(self, trade_date: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch and resample spot and options data for a single day.
        Returns: (resampled_spot_df, resampled_options_df, front_expiry)
        """
        start_time = datetime.now()
        conn = psycopg2.connect(**self.db_config)
        try:
            trade_dt = datetime.strptime(trade_date, '%Y-%m-%d')
            # Nifty session (approx UTC if DB stores UTC, but let's stick to the system's logic)
            # 03:45 UTC = 09:15 IST
            # 10:00 UTC = 15:30 IST
            # Let's use the Date/Time filters but and convert to timestamp range for index usage
            day_start = datetime.combine(trade_dt, datetime.min.time())
            day_end = day_start + timedelta(days=1)

            # 1. Get Front Expiry
            t0 = datetime.now()
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(expiry_date) FROM options WHERE date = %s AND expiry_date >= %s", (trade_date, trade_date))
            front_expiry = cursor.fetchone()[0]
            if not front_expiry:
                return None, None, None
            t1 = datetime.now()
            
            # 2. Fetch Spot Data (1-min) - Optimized Range Query
            spot_query = """
                SELECT timestamp AS ts, open, high, low, close
                FROM nifty
                WHERE timestamp >= %s AND timestamp < %s
                ORDER BY ts
            """
            # Note: We still need to filter hours IST precisely if the range is 24h
            # But the 'nifty' table usually only has session data
            spot_df = pd.read_sql(spot_query, conn, params=(day_start, day_end))
            t2 = datetime.now()
            if spot_df.empty:
                return None, None, None
            
            # Correct TZ handling: nifty table has IST (+05:30), options has None
            if spot_df['ts'].dt.tz is not None:
                # If it has TZ, convert to IST then make naive
                spot_df['ts'] = spot_df['ts'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            # Else if naive, we assume it's already IST wall-clock
            
            # Filter session hours
            spot_df = spot_df[(spot_df['ts'].dt.time >= datetime.strptime('09:15', '%H:%M').time()) & 
                               (spot_df['ts'].dt.time <= datetime.strptime('15:30', '%H:%M').time())].copy()
            
            if spot_df.empty:
                return None, None, None

            # 3. Calculate Strike Filter (±15 strikes from day move)
            s_min, s_max = spot_df['low'].min(), spot_df['high'].max()
            strike_min = int(np.floor(s_min / 50) * 50) - 750 # 15 strikes
            strike_max = int(np.ceil(s_max / 50) * 50) + 750
            
            # 4. Fetch Options Data - Filter by Strike Range
            options_query = """
                SELECT timestamp AS ts, strike, option_type, close as mkt_close, oi, iv, gamma, delta, volume
                FROM options
                WHERE date = %s AND expiry_date = %s
                  AND strike BETWEEN %s AND %s
                ORDER BY ts, strike, option_type
            """
            options_df = pd.read_sql(options_query, conn, params=(trade_date, str(front_expiry), strike_min, strike_max))
            t3 = datetime.now()
            
            # Options table is naive — assume it's already IST wall-clock
            if not options_df.empty:
                if options_df['ts'].dt.tz is not None:
                    options_df['ts'] = options_df['ts'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                
                options_df = options_df[(options_df['ts'].dt.time >= datetime.strptime('09:15', '%H:%M').time()) & 
                                         (options_df['ts'].dt.time <= datetime.strptime('15:30', '%H:%M').time())].copy()

            logging.info(f"  [DB] Fetch {trade_date}: Exp=0.0s, Spot={(t2-t1).total_seconds():.2f}s, Opt={(t3-t2).total_seconds():.2f}s | Strikes: {strike_min}-{strike_max}")

            # 4. Resample Spot to 3-min (OHLC)
            spot_df.set_index('ts', inplace=True)
            resampled_spot = spot_df.resample(self.interval).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).ffill()

            # 5. Resample Options to 3-min
            # We take the last snapshot of the options chain for each 3-min interval
            options_df.set_index('ts', inplace=True)
            
            # Group by timestamp (resampled) and strike/option_type
            resampled_options = options_df.groupby([pd.Grouper(freq=self.interval), 'strike', 'option_type']).last().reset_index()
            
            return resampled_spot, resampled_options, str(front_expiry)

        except Exception as e:
            logging.error(f"Error fetching data for {trade_date}: {e}")
            return None, None, None
        finally:
            conn.close()

    def get_day_start_oi(self, trade_date: str, front_expiry: str) -> Dict[Tuple[int, str], float]:
        """Fetch OI at 09:15 for incremental delta tracking."""
        conn = psycopg2.connect(**self.db_config)
        query = """
            SELECT strike, option_type, oi
            FROM options
            WHERE date = %s AND expiry_date = %s
              AND (timestamp AT TIME ZONE 'Asia/Kolkata')::time = '09:15:00'
        """
        cursor = conn.cursor()
        cursor.execute(query, (trade_date, front_expiry))
        day_start_oi = {(int(r[0]), r[1]): float(r[2]) if r[2] else 0.0 for r in cursor.fetchall()}
        conn.close()
        return day_start_oi
