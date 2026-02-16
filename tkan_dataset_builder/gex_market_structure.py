"""
5-Layer GEX Market Structure Analysis System

Layers:
    L1: Market Regime      — ±10 strikes from day-open ATM
    L2: ATM Environment    — ±4 strikes from current ATM
    L3: Gamma Landscape    — Peak, Trough, Flip (zero-crossing)
    L4: Strike-wise Profile — Full per-strike Net GEX breakdown
    L5: Directional Bias   — Upper vs Lower negative gamma intensity

Output: Integrated trade decision via L1 × L2 × L5 decision matrix.
"""

import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

from db_config import DB_CONFIG
from advanced_tree_models_optimized import StochasticVol2DTrinomialTree

# ============================================================================
# CONSTANTS
# ============================================================================

LOT_SIZE = 65
STRIKE_STEP = 50
REGIME_THRESHOLD = 500  # Millions — boundary between POSITIVE/NEGATIVE
RISK_FREE_RATE_CSV = (
    "/Users/prana/Desktop/working model/binomial_options_pricing/"
    "2d_vol_model_data/risk_free_rate.csv"
)


# ============================================================================
# DATA CLASSES — structured output for each layer
# ============================================================================

@dataclass
class StrikeGEX:
    """Per-strike GEX decomposition."""
    strike: int
    ce_gex: float = 0.0
    pe_gex: float = 0.0
    net_gex: float = 0.0
    dominant_side: str = "Balanced"  # "Call dominated" | "Put dominated" | "Balanced"


@dataclass
class LayerOneResult:
    """L1: Market Regime."""
    total_market_gex: float = 0.0
    regime: str = "NEUTRAL"
    strength: float = 0.0


@dataclass
class LayerTwoResult:
    """L2: ATM Environment."""
    atm_strike: int = 0
    atm_zone_strikes: List[int] = field(default_factory=list)
    long_count: int = 0
    short_count: int = 0
    environment: str = "MIXED GAMMA ZONE"
    atm_core_gex: float = 0.0


@dataclass
class LayerThreeResult:
    """L3: Gamma Landscape."""
    gamma_peak_strike: int = 0
    gamma_peak_value: float = 0.0
    gamma_trough_strike: int = 0
    gamma_trough_value: float = 0.0
    gamma_flip_strike: float = 0.0  # interpolated, can be float
    peak_relation: str = ""  # "ABOVE" | "BELOW" | "AT"
    trough_relation: str = ""
    distance_to_trough: float = 0.0


@dataclass
class LayerFourResult:
    """L4: Strike-wise Profile."""
    profile: List[StrikeGEX] = field(default_factory=list)
    upper_net_gex: float = 0.0
    lower_net_gex: float = 0.0


@dataclass
class LayerFiveResult:
    """L5: Directional Bias."""
    upper_negative_intensity: float = 0.0
    lower_negative_intensity: float = 0.0
    directional_asymmetry: float = 0.0
    bias_label: str = "Neutral / balanced"


@dataclass
class AnalysisResult:
    """Integrated output from all 5 layers."""
    timestamp: datetime = None
    spot: float = 0.0
    expiry_days: float = 0.0

    l1: LayerOneResult = field(default_factory=LayerOneResult)
    l2: LayerTwoResult = field(default_factory=LayerTwoResult)
    l3: LayerThreeResult = field(default_factory=LayerThreeResult)
    l4: LayerFourResult = field(default_factory=LayerFourResult)
    l5: LayerFiveResult = field(default_factory=LayerFiveResult)

    decision: str = ""
    best_sell_strike_ce: Optional[int] = None
    best_sell_strike_pe: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# DATA LAYER — DB fetching
# ============================================================================

def load_risk_free_rate(trade_date: str) -> float:
    """Load risk-free rate for the given date from CSV."""
    try:
        df = pd.read_csv(RISK_FREE_RATE_CSV)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        target = pd.to_datetime(trade_date)
        match = df[df["Date"].dt.date <= target.date()].sort_values("Date")
        if not match.empty:
            return float(match.iloc[-1]["Open"]) / 100.0
    except Exception as e:
        print(f"Warning: RFR load failed ({e}), using default 0.065")
    return 0.065


def get_front_expiry(conn, trade_date: str):
    """Return the nearest expiry date for the given trade date."""
    cur = conn.cursor()
    cur.execute(
        "SELECT MIN(expiry_date) FROM options WHERE date = %s AND expiry_date >= %s",
        (trade_date, trade_date),
    )
    row = cur.fetchone()
    cur.close()
    return row[0] if row and row[0] else None


def fetch_spot_data(conn, trade_date: str) -> pd.DataFrame:
    """Fetch 1-min Nifty spot bars for the trading session."""
    query = """
        SELECT
            (timestamp AT TIME ZONE 'Asia/Kolkata')::timestamp AS timestamp,
            close AS spot
        FROM nifty
        WHERE (timestamp AT TIME ZONE 'Asia/Kolkata')::date = %s
          AND (timestamp AT TIME ZONE 'Asia/Kolkata')::time >= '09:15:00'
          AND (timestamp AT TIME ZONE 'Asia/Kolkata')::time <= '15:30:00'
        ORDER BY timestamp
    """
    cur = conn.cursor()
    cur.execute(query, (trade_date,))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=["timestamp", "spot"])


def fetch_options_data(conn, trade_date: str, expiry, strikes) -> pd.DataFrame:
    """
    Fetch options chain for the given strikes on trade_date/expiry.

    Returns columns: timestamp, strike, option_type, mkt_close, oi, iv
    """
    query = """
        SELECT
            (o.timestamp AT TIME ZONE 'Asia/Kolkata')::timestamp AS timestamp,
            o.strike,
            o.option_type,
            o.mkt_close,
             o.oi,
             o.iv,
             o.volume
         FROM options o
        WHERE o.date = %s
          AND o.expiry_date = %s
          AND o.strike IN %s
          AND o.close > 0
          AND (o.timestamp AT TIME ZONE 'Asia/Kolkata')::time >= '09:15:00'
          AND (o.timestamp AT TIME ZONE 'Asia/Kolkata')::time <= '15:30:00'
        ORDER BY o.timestamp, o.strike, o.option_type
    """
    cur = conn.cursor()
    cur.execute(query, (trade_date, str(expiry), tuple(int(s) for s in strikes)))
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(
        rows,
        columns=["timestamp", "strike", "option_type", "mkt_close", "oi", "iv", "volume"],
    )


def fetch_day_start_oi(conn, trade_date: str, expiry) -> Dict[Tuple[int, str], float]:
    """Return {(strike, option_type): oi} at first timestamp of the day."""
    query = """
        SELECT strike, option_type, oi
        FROM options
        WHERE date = %s
          AND expiry_date = %s
          AND timestamp = (
              SELECT MIN(timestamp) FROM options
              WHERE date = %s AND expiry_date = %s
          )
          AND oi IS NOT NULL
    """
    cur = conn.cursor()
    cur.execute(query, (trade_date, str(expiry), trade_date, str(expiry)))
    result = {}
    for strike, otype, oi in cur.fetchall():
        result[(int(strike), otype)] = float(oi) if oi else 0.0
    cur.close()
    return result


# ============================================================================
# GEX CALCULATOR — per-strike GEX engine
# ============================================================================

class GEXCalculator:
    """
    Computes per-strike CE_GEX, PE_GEX, Net_GEX using:
        GEX(K) = OI_Delta(K) × sign(mispricing(K)) × Gamma(K) × Spot² × LotSize × bias_mult

    Where bias_mult is -1 for CE, +1 for PE (spec convention).
    """

    def __init__(self, model: StochasticVol2DTrinomialTree, r: float, n_steps: int = 30):
        self.model = model
        self.r = r
        self.n_steps = n_steps

    def compute_strike_gex(
        self,
        spot: float,
        T: float,
        t_now: datetime,
        strikes: np.ndarray,
        option_types: np.ndarray,
        mkt_prices: np.ndarray,
        ivs: np.ndarray,
        oi_deltas: np.ndarray,
        absolute_oi: np.ndarray = None,
        mm_inventory: np.ndarray = None,
        volumes: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Vectorised per-strike GEX calculation.

        Returns DataFrame with columns:
            strike, option_type, gamma, charm, vanna, mispricing, mm_sign, oi_delta,
            ce_gex, pe_gex, net_gex_contribution, turnover
        """
        # Base IV for model
        valid_ivs = ivs[ivs > 0]
        base_iv = float(np.nanmean(valid_ivs)) if len(valid_ivs) > 0 else 0.15
        clean_ivs = np.where((ivs <= 0) | np.isnan(ivs), base_iv, ivs)

        # n_steps calculation
        n_steps = max(5, min(int(T * 365), self.n_steps))

        # Use extended greeks calculation for charm/vanna
        # Note: We need to import it here or at top
        try:
            from advanced_tree_models_optimized import calculate_greeks_extended_batch
            greeks = calculate_greeks_extended_batch(
                self.model, np.array([spot]), strikes, T, t_now, self.r, 
                np.array([base_iv]), clean_ivs, n_steps, option_types
            )
            gamma = greeks['gamma']
            charm = greeks['charm']
            vanna = greeks['vanna']
            # We still need p_model for mispricing
            p_model = self.model.price_batch(
                spot, strikes, T, t_now, self.r, base_iv, clean_ivs, n_steps, list(option_types)
            )
        except (ImportError, KeyError):
            # Fallback to base pricing if extended not found
            p_model = self.model.price_batch(
                spot, strikes, T, t_now, self.r, base_iv, clean_ivs, n_steps, list(option_types)
            )
            h = spot * 0.01
            p_up = self.model.price_batch(
                spot + h, strikes, T, t_now, self.r, base_iv, clean_ivs, n_steps, list(option_types)
            )
            p_dn = self.model.price_batch(
                spot - h, strikes, T, t_now, self.r, base_iv, clean_ivs, n_steps, list(option_types)
            )
            gamma = (p_up - 2 * p_model + p_dn) / (h ** 2)
            charm = np.zeros_like(gamma)
            vanna = np.zeros_like(gamma)

        # Mispricing & MM sign
        mispricing = mkt_prices - p_model
        mm_sign = np.sign(mispricing)

        # Bias multiplier: CE → -1, PE → +1
        bias_mult = np.where(option_types == "CE", -1.0, 1.0)

        # Per-option GEX contribution
        qty_for_gex = mm_inventory if mm_inventory is not None else oi_deltas
        
        if mm_inventory is not None:
            gex = mm_inventory * gamma * (spot ** 2) * 0.01 * bias_mult
        else:
            gex = oi_deltas * mm_sign * gamma * (spot ** 2) * 0.01 * bias_mult

        # Net Sentiment
        if absolute_oi is not None:
            sentiment_qty = mm_inventory if mm_inventory is not None else (oi_deltas * mm_sign)
            sentiment = sentiment_qty * bias_mult
        else:
            sentiment = np.zeros_like(gex)

        # Heatmap GEX
        if absolute_oi is not None:
            heatmap_qty = mm_inventory if mm_inventory is not None else (absolute_oi * mm_sign)
            heatmap_gex = gamma * (spot ** 2) * 0.01 * heatmap_qty
        else:
            heatmap_gex = gex  # fallback

        # Turnover Logic
        if volumes is not None:
            turnover = np.abs(oi_deltas) / (volumes + 1e-9)
        else:
            turnover = np.zeros_like(gex)

        return pd.DataFrame({
            "strike": strikes,
            "option_type": option_types,
            "gamma": gamma,
            "charm": charm,
            "vanna": vanna,
            "mispricing": mispricing,
            "mm_sign": mm_sign,
            "oi_delta": oi_deltas,
            "gex": gex,
            "sentiment": sentiment,
            "heatmap_gex": heatmap_gex,
            "turnover": turnover
        })


# ============================================================================
# MARKET STRUCTURE ANALYZER — 5-layer stack
# ============================================================================

class MarketStructureAnalyzer:
    """Runs the 5-layer GEX analysis stack on pre-computed per-strike GEX data."""

    def __init__(self, open_atm: int):
        """
        Parameters:
            open_atm: ATM strike at day open (anchors Layer 1).
        """
        self.open_atm = open_atm

    # ------------------------------------------------------------------ L1
    def layer1_market_regime(self, gex_df: pd.DataFrame) -> LayerOneResult:
        """
        Calculate total Net GEX from ±10 strikes around day-open ATM.

        Parameters:
            gex_df: DataFrame with columns [strike, option_type, gex]

        Returns:
            LayerOneResult with regime classification.
        """
        lo = self.open_atm - 10 * STRIKE_STEP
        hi = self.open_atm + 10 * STRIKE_STEP
        zone = gex_df[(gex_df["strike"] >= lo) & (gex_df["strike"] <= hi)]

        # Net GEX per strike (CE + PE combined) — using heatmap_gex for consistency
        col = "heatmap_gex" if "heatmap_gex" in gex_df.columns else "gex"
        strike_net = zone.groupby("strike")[col].sum()
        total = float(strike_net.sum()) / 1e6  # in millions

        # Classify
        if total > REGIME_THRESHOLD:
            regime = "POSITIVE GEX"
        elif total < -REGIME_THRESHOLD:
            regime = "NEGATIVE GEX"
        else:
            regime = "NEUTRAL GEX"

        strength = min(abs(total) / 1000.0 * 100, 100.0)

        return LayerOneResult(
            total_market_gex=total,
            regime=regime,
            strength=round(strength, 1),
        )

    # ------------------------------------------------------------------ L2
    def layer2_atm_environment(
        self, gex_df: pd.DataFrame, current_spot: float
    ) -> LayerTwoResult:
        """
        Analyse ±4 strikes around current ATM.

        Parameters:
            gex_df: DataFrame with columns [strike, option_type, gex]
            current_spot: Current Nifty spot.

        Returns:
            LayerTwoResult with ATM zone classification.
        """
        current_atm = int(round(current_spot / STRIKE_STEP) * STRIKE_STEP)
        zone_strikes = [current_atm + i * STRIKE_STEP for i in range(-4, 5)]

        zone = gex_df[gex_df["strike"].isin(zone_strikes)]
        col = "heatmap_gex" if "heatmap_gex" in gex_df.columns else "gex"
        strike_net = zone.groupby("strike")[col].sum()

        long_count = int((strike_net > 0).sum())
        short_count = int((strike_net < 0).sum())

        if long_count >= 6:
            env = "LONG GAMMA ZONE"
        elif short_count >= 6:
            env = "SHORT GAMMA ZONE"
        else:
            env = "MIXED GAMMA ZONE"

        atm_core = float(strike_net.sum()) / 1e6

        return LayerTwoResult(
            atm_strike=current_atm,
            atm_zone_strikes=zone_strikes,
            long_count=long_count,
            short_count=short_count,
            environment=env,
            atm_core_gex=round(atm_core, 2),
        )

    # ------------------------------------------------------------------ L3
    def layer3_gamma_landscape(
        self, gex_df: pd.DataFrame, current_spot: float
    ) -> LayerThreeResult:
        """
        Find Gamma Peak, Trough, and Flip (zero-crossing) strikes.

        Uses the 'sentiment' column to match zone detection in the GEX
        heatmap visualization (strike_oi_replay_visualization.py).

        Parameters:
            gex_df: DataFrame with columns [strike, option_type, sentiment/gex]
            current_spot: Current Nifty spot.

        Returns:
            LayerThreeResult with peak/trough/flip data.
        """
        col = "heatmap_gex" if "heatmap_gex" in gex_df.columns else "gex"
        strike_net = gex_df.groupby("strike")[col].sum().sort_index()

        if strike_net.empty:
            return LayerThreeResult()

        strikes_arr = strike_net.index.values
        vals = strike_net.values / 1e6  # in millions

        # Peak — highest positive
        peak_idx = np.argmax(vals)
        peak_strike = int(strikes_arr[peak_idx])
        peak_val = float(vals[peak_idx])

        # Trough — most negative
        trough_idx = np.argmin(vals)
        trough_strike = int(strikes_arr[trough_idx])
        trough_val = float(vals[trough_idx])

        # Flip — zero-crossing closest to current spot
        flip_strike = 0.0
        min_dist = float("inf")
        for i in range(len(vals) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]) and vals[i + 1] != vals[i]:
                # Linear interpolation
                z = strikes_arr[i] - vals[i] * (
                    strikes_arr[i + 1] - strikes_arr[i]
                ) / (vals[i + 1] - vals[i])
                dist = abs(z - current_spot)
                if dist < min_dist:
                    min_dist = dist
                    flip_strike = float(z)

        # Relations to spot
        def _relation(strike_val):
            diff = strike_val - current_spot
            if abs(diff) <= STRIKE_STEP:
                return "AT"
            return "ABOVE" if diff > 0 else "BELOW"

        return LayerThreeResult(
            gamma_peak_strike=peak_strike,
            gamma_peak_value=round(peak_val, 2),
            gamma_trough_strike=trough_strike,
            gamma_trough_value=round(trough_val, 2),
            gamma_flip_strike=round(flip_strike, 1),
            peak_relation=_relation(peak_strike),
            trough_relation=_relation(trough_strike),
            distance_to_trough=abs(trough_strike - current_spot),
        )

    # ------------------------------------------------------------------ L4
    def layer4_strike_profile(
        self, gex_df: pd.DataFrame, current_spot: float
    ) -> LayerFourResult:
        """
        Full strike-by-strike Net Sentiment profile split into upper/lower.

        Uses the 'sentiment' column (oi × sign(mispricing) × bias_mult) to
        match the Net Strike Sentiment display in strike_oi_replay_visualization.

        Parameters:
            gex_df: DataFrame with columns [strike, option_type, sentiment]
            current_spot: Current Nifty spot.

        Returns:
            LayerFourResult with per-strike breakdown.
        """
        col = "sentiment" if "sentiment" in gex_df.columns else "gex"
        strike_net = gex_df.groupby("strike")[col].sum().sort_index()

        # Per-strike decomposition
        ce_gex = gex_df[gex_df["option_type"] == "CE"].groupby("strike")[col].sum()
        pe_gex = gex_df[gex_df["option_type"] == "PE"].groupby("strike")[col].sum()

        profile = []
        for strike in strike_net.index:
            ce_val = float(ce_gex.get(strike, 0))
            pe_val = float(pe_gex.get(strike, 0))
            net = ce_val + pe_val

            # Dominance check (within 20% = balanced)
            total_abs = abs(ce_val) + abs(pe_val)
            if total_abs > 0:
                ratio = abs(abs(ce_val) - abs(pe_val)) / total_abs
                if ratio < 0.2:
                    dom = "Balanced"
                elif abs(ce_val) > abs(pe_val):
                    dom = "Call dominated"
                else:
                    dom = "Put dominated"
            else:
                dom = "Balanced"

            profile.append(StrikeGEX(
                strike=int(strike),
                ce_gex=round(ce_val / 1e6, 2),
                pe_gex=round(pe_val / 1e6, 2),
                net_gex=round(net / 1e6, 2),
                dominant_side=dom,
            ))

        # Upper/Lower aggregation
        upper_total = sum(
            s.net_gex for s in profile if s.strike > current_spot + STRIKE_STEP
        )
        lower_total = sum(
            s.net_gex for s in profile if s.strike < current_spot - STRIKE_STEP
        )

        return LayerFourResult(
            profile=profile,
            upper_net_gex=round(upper_total, 2),
            lower_net_gex=round(lower_total, 2),
        )

    # ------------------------------------------------------------------ L5
    def layer5_directional_bias(
        self, l4: LayerFourResult, current_spot: float
    ) -> LayerFiveResult:
        """
        Compute directional asymmetry from negative gamma intensity.

        Parameters:
            l4: LayerFourResult (pre-computed strike profile).
            current_spot: Current Nifty spot.

        Returns:
            LayerFiveResult with asymmetry score.
        """
        upper_neg = sum(
            abs(s.net_gex)
            for s in l4.profile
            if s.strike > current_spot + STRIKE_STEP and s.net_gex < 0
        )
        lower_neg = sum(
            abs(s.net_gex)
            for s in l4.profile
            if s.strike < current_spot - STRIKE_STEP and s.net_gex < 0
        )

        denom = upper_neg + lower_neg + 1e-9  # avoid div-by-zero
        asymmetry = (upper_neg - lower_neg) / denom

        # Label
        if asymmetry >= 0.5:
            label = "Strong upside breakout bias"
        elif asymmetry >= 0.2:
            label = "Mild upside bias"
        elif asymmetry <= -0.5:
            label = "Strong downside breakout bias"
        elif asymmetry <= -0.2:
            label = "Mild downside bias"
        else:
            label = "Neutral / balanced"

        return LayerFiveResult(
            upper_negative_intensity=round(upper_neg, 2),
            lower_negative_intensity=round(lower_neg, 2),
            directional_asymmetry=round(asymmetry, 3),
            bias_label=label,
        )


# ============================================================================
# DECISION ENGINE — L1 × L2 × L5 cross-reference
# ============================================================================

class DecisionEngine:
    """Maps the 3-dimensional regime space to actionable trade decisions."""

    @staticmethod
    def decide(
        l1: LayerOneResult,
        l2: LayerTwoResult,
        l3: LayerThreeResult,
        l4: LayerFourResult,
        l5: LayerFiveResult,
        spot: float,
    ) -> Tuple[str, Optional[int], Optional[int], List[str]]:
        """
        Cross-reference all layers via the decision matrix.

        Returns:
            (decision_text, best_sell_ce_strike, best_sell_pe_strike, warnings)
        """
        warnings_list = []

        # Proximity to trough warning
        if l3.distance_to_trough < 100:
            warnings_list.append(
                f"ALERT: Spot {spot:.0f} is {l3.distance_to_trough:.0f} pts from "
                f"gamma trough at {l3.gamma_trough_strike}. Avoid naked selling."
            )

        # Proximity to flip warning
        if l3.gamma_flip_strike > 0:
            dist_to_flip = abs(spot - l3.gamma_flip_strike)
            if dist_to_flip < 100:
                warnings_list.append(
                    f"Spot approaching gamma flip at {l3.gamma_flip_strike:.0f} "
                    f"({dist_to_flip:.0f} pts away). Regime boundary transition."
                )

        # Determine upper/lower gamma character for decision matrix
        upper_all_neg = all(
            s.net_gex < 0
            for s in l4.profile
            if s.strike > spot + STRIKE_STEP
        )
        lower_all_neg = all(
            s.net_gex < 0
            for s in l4.profile
            if s.strike < spot - STRIKE_STEP
        )
        both_neg = upper_all_neg and lower_all_neg
        both_pos = (
            all(s.net_gex >= 0 for s in l4.profile if s.strike > spot + STRIKE_STEP)
            and all(s.net_gex >= 0 for s in l4.profile if s.strike < spot - STRIKE_STEP)
        )

        # Best sell strikes — find strongest positive GEX strikes
        best_ce = None
        best_pe = None
        above = [s for s in l4.profile if s.strike > spot + STRIKE_STEP and s.net_gex > 0]
        below = [s for s in l4.profile if s.strike < spot - STRIKE_STEP and s.net_gex > 0]
        if above:
            best_ce = max(above, key=lambda s: s.net_gex).strike
        if below:
            best_pe = max(below, key=lambda s: s.net_gex).strike

        # Decision matrix — resolve MIXED zone by ATM core GEX sign
        is_pos = "POSITIVE" in l1.regime
        is_neg = "NEGATIVE" in l1.regime
        is_neutral = "NEUTRAL" in l1.regime

        # For L2: MIXED zone → proxy to LONG/SHORT via ATM core GEX sign
        if "LONG" in l2.environment:
            is_long, is_short = True, False
        elif "SHORT" in l2.environment:
            is_long, is_short = False, True
        else:
            # MIXED → use ATM core GEX sign as tie-breaker
            is_long = l2.atm_core_gex >= 0
            is_short = l2.atm_core_gex < 0

        if is_neutral:
            decision = "WAIT for regime clarity — neutral GEX"
            best_ce, best_pe = None, None
        elif is_neg and is_short and both_neg:
            decision = "AVOID ALL NAKED POSITIONS — explosive environment"
            best_ce, best_pe = None, None
        elif is_neg and is_short:
            if l5.directional_asymmetry > 0.2:
                decision = "STRONG UPSIDE BREAKOUT expected — buy CE spread"
                best_ce = None
            elif l5.directional_asymmetry < -0.2:
                decision = "STRONG DOWNSIDE BREAK expected — buy PE spread"
                best_pe = None
            else:
                decision = "EXPLOSIVE: Whichever direction triggers first will run far"
                best_ce, best_pe = None, None
        elif is_neg and is_long:
            if l5.directional_asymmetry > 0.2:
                decision = "BUY CALL SPREAD — avoid CE selling"
                best_ce = None
            elif l5.directional_asymmetry < -0.2:
                decision = "BUY PUT SPREAD — avoid PE selling"
                best_pe = None
            else:
                decision = "NEGATIVE regime but LONG ATM — reduce size, use spreads"
        elif is_pos and is_short:
            decision = "REDUCE SIZE — use spreads only (POSITIVE regime but SHORT ATM)"
        elif is_pos and is_long:
            if both_pos:
                decision = "PINNED — best conditions for strangle/straddle selling"
            elif l5.directional_asymmetry > 0.2:
                decision = "SELL PE at support — upside bias"
                best_ce = None
            elif l5.directional_asymmetry < -0.2:
                decision = "SELL CE at resistance — downside bias"
                best_pe = None
            else:
                decision = "SELL STRANGLE at Peak/Trough levels"

        return decision, best_ce, best_pe, warnings_list


# ============================================================================
# ORCHESTRATOR — runs full analysis for a single timestamp
# ============================================================================

def analyze_timestamp(
    gex_calc: GEXCalculator,
    analyzer: MarketStructureAnalyzer,
    spot: float,
    T: float,
    t_now: datetime,
    bar_df: pd.DataFrame,
    day_start_oi: Dict,
    mm_inventory: Dict[Tuple[int, str], float] = None,
    prev_oi: Dict[Tuple[int, str], float] = None,
) -> Tuple[AnalysisResult, Dict[Tuple[int, str], float]]:
    """
    Run the complete 5-layer stack on a single timestamp bar with Inventory tracking.

    Parameters:
        gex_calc: Initialised GEXCalculator.
        analyzer: Initialised MarketStructureAnalyzer (with open_atm set).
        spot: Current spot price.
        T: Time to expiry in years.
        t_now: Current timestamp.
        bar_df: Options data for this timestamp.
        day_start_oi: {(strike, option_type): oi} at day start.
        mm_inventory: (strike, type) -> current quantity (signed).
        prev_oi: (strike, type) -> absolute OI at previous check.

    Returns:
        (AnalysisResult, updated_mm_inventory)
    """
    if bar_df.empty:
        return AnalysisResult(timestamp=t_now, spot=spot), (mm_inventory or {})

    # Step 1: Preliminary price/gamma/mispricing to get MM signs
    # To determine MM sign for the DELTA OI flow.
    strikes = bar_df["strike"].values.astype(float)
    types = bar_df["option_type"].values
    mkt_prices = bar_df["mkt_close"].values.astype(float)
    ivs = bar_df["iv"].values.astype(float)
    absolute_oi = bar_df["oi"].values.astype(float)

    # Temporary GEX calc to get mispricing
    # (We need mkt_prices vs model pricing at this instant)
    valid_ivs = ivs[ivs > 0]
    base_iv = float(np.nanmean(valid_ivs)) if len(valid_ivs) > 0 else 0.15
    clean_ivs = np.where((ivs <= 0) | np.isnan(ivs), base_iv, ivs)
    
    n_steps = max(5, min(int(T * 365), gex_calc.n_steps))
    p_model = gex_calc.model.price_batch(
        spot, strikes, T, t_now, gex_calc.r, base_iv, clean_ivs, n_steps, list(types)
    )
    mm_signs = np.sign(mkt_prices - p_model)
    mm_sign_map = {(int(s), t): sign for s, t, sign in zip(strikes, types, mm_signs)}

    # Step 2: Update MM Inventory
    if mm_inventory is None:
        mm_inventory = {}

    current_time = t_now.strftime("%H:%M")
    
    # 09:18 Build
    if not mm_inventory and current_time >= "09:18":
        for s, t, o in zip(strikes, types, absolute_oi):
            sign = mm_sign_map.get((int(s), t), 0)
            mm_inventory[(int(s), t)] = o * sign
    elif mm_inventory and current_time > "09:18":
        # Update based on Delta OI
        for s, t, o in zip(strikes, types, absolute_oi):
            key = (int(s), t)
            prev_o = prev_oi.get(key, o) if prev_oi else o
            delta_oi = o - prev_o
            
            # If OI increased, sign it with current mispricing
            # If OI decreased, we proportionally reduce existing inventory
            if delta_oi > 0:
                sign = mm_sign_map.get(key, 0)
                mm_inventory[key] = mm_inventory.get(key, 0) + (delta_oi * sign)
            elif delta_oi < 0:
                # Pro-rata reduction of whatever sign we had
                curr_inv = mm_inventory.get(key, 0)
                if abs(o) < 1e-9:
                    mm_inventory[key] = 0
                else:
                    # Scale down: new_inv = old_inv * (new_total_oi / old_total_oi)
                    mm_inventory[key] = curr_inv * (o / prev_o)

    # Prepare inventory array for vectorized calc
    inv_arr = np.array([mm_inventory.get((int(s), t), 0.0) for s, t in zip(strikes, types)])
    
    # OI Delta (legacy/intraday only)
    oi_deltas = bar_df.apply(
        lambda row: float(row["oi"]) - day_start_oi.get(
            (int(row["strike"]), row["option_type"]), 0.0
        )
        if pd.notna(row["oi"]) else 0.0,
        axis=1,
    ).values

    # Step 3: Compute final GEX using inventory
    gex_df = gex_calc.compute_strike_gex(
        spot, T, t_now, strikes, types, mkt_prices, ivs, oi_deltas, absolute_oi, mm_inventory=inv_arr
    )

    # Layers 1-5
    l1 = analyzer.layer1_market_regime(gex_df)
    l2 = analyzer.layer2_atm_environment(gex_df, spot)
    l3 = analyzer.layer3_gamma_landscape(gex_df, spot)
    l4 = analyzer.layer4_strike_profile(gex_df, spot)
    l5 = analyzer.layer5_directional_bias(l4, spot)

    decision, best_ce, best_pe, warns = DecisionEngine.decide(l1, l2, l3, l4, l5, spot)

    res = AnalysisResult(
        timestamp=t_now,
        spot=spot,
        expiry_days=T * 365,
        l1=l1,
        l2=l2,
        l3=l3,
        l4=l4,
        l5=l5,
        decision=decision,
        best_sell_strike_ce=best_ce,
        best_sell_strike_pe=best_pe,
        warnings=warns,
    )
    return res, mm_inventory


# ============================================================================
# FULL-DAY ANALYSIS — iterate over all (or sampled) timestamps
# ============================================================================

def run_full_day_analysis(
    trade_date: str,
    interval_minutes: int = 5,
    specific_time: str = None,
) -> List[AnalysisResult]:
    """
    Run the 5-layer analysis for an entire trading day.

    Parameters:
        trade_date: YYYY-MM-DD.
        interval_minutes: How often to compute (default 5 min).
        specific_time: If set (HH:MM), only analyse this single timestamp.

    Returns:
        List of AnalysisResult, one per analysed timestamp.
    """
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        # Metadata
        r = load_risk_free_rate(trade_date)
        expiry = get_front_expiry(conn, trade_date)
        if expiry is None:
            print(f"ERROR: No expiry found for {trade_date}")
            return []

        expiry_ts = pd.to_datetime(str(expiry)) + timedelta(hours=15, minutes=30)
        print(f"Trade date: {trade_date} | Expiry: {expiry} | RFR: {r:.4f}")

        # Spot data
        spot_df = fetch_spot_data(conn, trade_date)
        if spot_df.empty:
            print("ERROR: No spot data")
            return []

        # Open ATM (anchored to day start)
        open_spot = float(spot_df.iloc[0]["spot"])
        open_atm = int(round(open_spot / STRIKE_STEP) * STRIKE_STEP)
        print(f"Open spot: {open_spot:.2f} | Open ATM: {open_atm}")

        # Define strike universe — broad enough for ±10 from open_atm and also
        # to cover spot movement during the day
        min_spot = spot_df["spot"].min()
        max_spot = spot_df["spot"].max()
        strike_lo = min(open_atm - 10 * STRIKE_STEP, int(min_spot / STRIKE_STEP) * STRIKE_STEP - 5 * STRIKE_STEP)
        strike_hi = max(open_atm + 10 * STRIKE_STEP, int(max_spot / STRIKE_STEP) * STRIKE_STEP + 5 * STRIKE_STEP)
        all_strikes = list(range(strike_lo, strike_hi + STRIKE_STEP, STRIKE_STEP))

        print(f"Strike universe: {strike_lo} to {strike_hi} ({len(all_strikes)} strikes)")

        # Fetch options data (all at once for efficiency)
        opts_df = fetch_options_data(conn, trade_date, expiry, all_strikes)
        if opts_df.empty:
            print("ERROR: No options data")
            return []

        # Day-start OI
        day_oi = fetch_day_start_oi(conn, trade_date, expiry)
        print(f"Day-start OI cached for {len(day_oi)} strike-type pairs")

        # Initialise components
        model = StochasticVol2DTrinomialTree()
        gex_calc = GEXCalculator(model, r)
        analyzer = MarketStructureAnalyzer(open_atm)

        # Determine timestamps to analyse
        timestamps = sorted(opts_df["timestamp"].unique())

        if specific_time:
            # Filter to single timestamp closest to requested time
            target = pd.to_datetime(f"{trade_date} {specific_time}")
            timestamps = [min(timestamps, key=lambda t: abs((pd.to_datetime(t) - target).total_seconds()))]
            print(f"Analysing single timestamp: {timestamps[0]}")
        elif interval_minutes > 1:
            # Subsample
            ts_series = pd.to_datetime(pd.Series(timestamps))
            mask = (ts_series - ts_series.iloc[0]).dt.total_seconds() % (interval_minutes * 60) < 60
            timestamps = [timestamps[i] for i in range(len(timestamps)) if mask.iloc[i]]
            print(f"Analysing {len(timestamps)} timestamps at {interval_minutes}-min intervals")
        else:
            print(f"Analysing all {len(timestamps)} timestamps")

        results = []
        for i, ts in enumerate(timestamps):
            if i % 10 == 0 and len(timestamps) > 10:
                print(f"  Progress: {i}/{len(timestamps)}")

            ts_dt = pd.to_datetime(ts)
            bar = opts_df[opts_df["timestamp"] == ts].copy()

            # Spot at this timestamp
            spot_row = spot_df[spot_df["timestamp"] <= ts]
            if spot_row.empty:
                continue
            spot = float(spot_row.iloc[-1]["spot"])

            T = max((expiry_ts - ts_dt).total_seconds() / (365.25 * 24 * 3600), 1 / 365)

            result = analyze_timestamp(
                gex_calc, analyzer, spot, T, ts_dt, bar, day_oi
            )
            results.append(result)

        print(f"  Completed: {len(results)} analysis snapshots\n")
        return results

    finally:
        conn.close()


# ============================================================================
# FORMATTED OUTPUT — matches spec's sample output
# ============================================================================

def print_analysis(result: AnalysisResult):
    """Print the full analysis output for a single timestamp, matching spec format."""
    r = result
    l1, l2, l3, l4, l5 = r.l1, r.l2, r.l3, r.l4, r.l5

    print("═" * 55)
    print("GEX MARKET STRUCTURE ANALYSIS")
    ts_str = r.timestamp.strftime("%H:%M") if r.timestamp else "N/A"
    print(f"Time: {ts_str} | Spot: {r.spot:,.0f} | Expiry: {r.expiry_days:.1f} days")
    print("═" * 55)

    # L1
    print(f"\nLAYER 1: MARKET REGIME")
    print("─" * 48)
    print(f"  Total Market GEX:   {l1.total_market_gex:+,.0f}M")
    print(f"  Regime:             {l1.regime} (Strength: {l1.strength:.0f}/100)")
    if "POSITIVE" in l1.regime:
        print("  Implication:        Mean-reverting, range-bound expected")
    elif "NEGATIVE" in l1.regime:
        print("  Implication:        Trending/breakout environment")
    else:
        print("  Implication:        Balanced, no structural bias")

    # L2
    print(f"\nLAYER 2: ATM ENVIRONMENT ({l2.atm_zone_strikes[0]} – {l2.atm_zone_strikes[-1]})")
    print("─" * 48)
    print(f"  Long Gamma Strikes: {l2.long_count} of 9")
    print(f"  Short Gamma Strikes:{l2.short_count} of 9")
    print(f"  ATM Core GEX:       {l2.atm_core_gex:+,.0f}M")
    print(f"  Environment:        {l2.environment}")

    # L3
    print(f"\nLAYER 3: GAMMA LANDSCAPE")
    print("─" * 48)
    print(f"  Gamma Peak:         {l3.gamma_peak_strike} ({l3.gamma_peak_value:+,.0f}M) ← {l3.peak_relation} spot")
    print(f"  Gamma Trough:       {l3.gamma_trough_strike} ({l3.gamma_trough_value:+,.0f}M) ← {l3.trough_relation} spot")
    if l3.gamma_flip_strike > 0:
        print(f"  Gamma Flip:         {l3.gamma_flip_strike:,.0f} ← Regime boundary")

    # L4
    print(f"\nLAYER 4: STRIKE-WISE PROFILE")
    print("─" * 48)
    upper = [s for s in l4.profile if s.strike > r.spot + STRIKE_STEP]
    lower = [s for s in l4.profile if s.strike < r.spot - STRIKE_STEP]
    near  = [s for s in l4.profile if abs(s.strike - r.spot) <= STRIKE_STEP]

    if upper:
        print(f"  UPPER STRIKES (Above {r.spot:,.0f}):")
        for s in sorted(upper, key=lambda x: x.strike):
            tag = "NEGATIVE" if s.net_gex < 0 else "Positive"
            extra = ""
            if s.strike == l3.gamma_trough_strike:
                extra = " ← Trough"
            elif s.strike == l3.gamma_peak_strike:
                extra = " ← Peak"
            print(f"    {s.strike:>6}: Net GEX = {s.net_gex:+8,.0f}M [{tag}]{extra}")

    if near:
        print(f"  ──── SPOT: {r.spot:,.0f} ────")

    if lower:
        print(f"  LOWER STRIKES (Below {r.spot:,.0f}):")
        for s in sorted(lower, key=lambda x: -x.strike):
            tag = "NEGATIVE" if s.net_gex < 0 else "Positive"
            extra = ""
            if s.strike == l3.gamma_trough_strike:
                extra = " ← Trough"
            elif s.strike == l3.gamma_peak_strike:
                extra = " ← Peak"
            print(f"    {s.strike:>6}: Net GEX = {s.net_gex:+8,.0f}M [{tag}]{extra}")

    # L5
    print(f"\nLAYER 5: DIRECTIONAL BIAS")
    print("─" * 48)
    upper_neg_count = sum(1 for s in upper if s.net_gex < 0)
    lower_neg_count = sum(1 for s in lower if s.net_gex < 0)
    print(f"  Upper Negative:     {l5.upper_negative_intensity:,.0f}M ({upper_neg_count} of {len(upper)} strikes)")
    print(f"  Lower Negative:     {l5.lower_negative_intensity:,.0f}M ({lower_neg_count} of {len(lower)} strikes)")
    print(f"  Asymmetry:          {l5.directional_asymmetry:+.3f} ({l5.bias_label})")

    # Decision
    print(f"\n{'═' * 55}")
    print("INTEGRATED DECISION")
    print("═" * 55)
    print(f"  Regime:      {l1.regime} + {l2.environment}")
    print(f"  Direction:   {l5.bias_label} (Score: {l5.directional_asymmetry:+.2f})")
    print(f"\n  >>> {r.decision}")

    if r.best_sell_strike_ce:
        ce_s = next((s for s in l4.profile if s.strike == r.best_sell_strike_ce), None)
        ce_val = f" ({ce_s.net_gex:+,.0f}M)" if ce_s else ""
        print(f"  Sell CE:     {r.best_sell_strike_ce}{ce_val}")
    else:
        print(f"  Sell CE:     ✗ AVOID")

    if r.best_sell_strike_pe:
        pe_s = next((s for s in l4.profile if s.strike == r.best_sell_strike_pe), None)
        pe_val = f" ({pe_s.net_gex:+,.0f}M)" if pe_s else ""
        print(f"  Sell PE:     {r.best_sell_strike_pe}{pe_val}")
    else:
        print(f"  Sell PE:     ✗ AVOID")

    for w in r.warnings:
        print(f"\n  ⚠ {w}")

    print("═" * 55 + "\n")


# ============================================================================
# RESULT → DICT (for CSV / programmatic use)
# ============================================================================

def result_to_dict(r: AnalysisResult) -> Dict:
    """Convert an AnalysisResult to a flat dict of the 17 key output variables."""
    return {
        "timestamp": r.timestamp,
        "spot": r.spot,
        "total_market_gex": r.l1.total_market_gex,
        "regime": r.l1.regime,
        "regime_strength": r.l1.strength,
        "atm_environment": r.l2.environment,
        "atm_core_gex": r.l2.atm_core_gex,
        "gamma_peak_strike": r.l3.gamma_peak_strike,
        "gamma_peak_value": r.l3.gamma_peak_value,
        "gamma_trough_strike": r.l3.gamma_trough_strike,
        "gamma_trough_value": r.l3.gamma_trough_value,
        "gamma_flip_strike": r.l3.gamma_flip_strike,
        "upper_negative_intensity": r.l5.upper_negative_intensity,
        "lower_negative_intensity": r.l5.lower_negative_intensity,
        "directional_asymmetry": r.l5.directional_asymmetry,
        "decision": r.decision,
        "best_sell_strike_ce": r.best_sell_strike_ce,
        "best_sell_strike_pe": r.best_sell_strike_pe,
    }


# ============================================================================
# ZONE-ALIGNED POSITION SIGNAL
# ============================================================================
# Uses gamma zones (same logic as strike_oi_replay_visualization.py) to
# identify the closest structural level, then reads net strike sentiment
# above/below spot to decide position direction.
#
# Simple rules:
#   - Positive net below spot → Support → bias is UP → sell PE near peak
#   - Negative net above spot → Breakout risk → bias is UP → buy CE near zone
#   - Negative net below spot → Breakdown risk → bias is DOWN → buy PE near zone
#   - Positive net above spot → Resistance → bias is DOWN → sell CE near peak
# ============================================================================

@dataclass
class ZoneSignal:
    """Output from zone-aligned position logic."""
    timestamp: datetime = None
    spot: float = 0.0

    # Zones detected
    nearest_zone_type: str = ""    # "peak" | "trough" | "zero"
    nearest_zone_strike: float = 0.0
    nearest_zone_distance: float = 0.0

    all_peaks: List[float] = field(default_factory=list)
    all_troughs: List[float] = field(default_factory=list)
    all_zeros: List[float] = field(default_factory=list)

    # Net sentiment split
    lower_net_positive: bool = False   # Below spot: net positive = support
    upper_net_negative: bool = False   # Above spot: net negative = breakout

    # Signal
    bias: str = ""           # "BULLISH" | "BEARISH" | "NEUTRAL"
    action: str = ""         # e.g. "SELL PE near 25400", "BUY CE near 25750"
    zone_strike: int = 0     # Recommended strike for position
    zone_type: str = ""      # "sell_pe" | "sell_ce" | "buy_ce" | "buy_pe" | "none"


def get_zone_position_signal(result: AnalysisResult) -> ZoneSignal:
    """
    Derive a simple directional position signal from gamma zones + net sentiment.

    Uses the same zone-detection approach as strike_oi_replay_visualization.py:
    - Peaks: strikes with max positive Net GEX (support / gamma walls)
    - Troughs: strikes with max negative Net GEX (acceleration pockets)
    - Zeros: sign-flip strikes (regime boundaries)

    Then checks net sentiment above/below spot to determine bias:
    - Positive net GEX below spot = support exists = bullish
    - Negative net GEX above spot = breakout zone = bullish
    - Vice versa = bearish

    Parameters:
        result: AnalysisResult from the 5-layer analysis.

    Returns:
        ZoneSignal with bias, action, and recommended zone strike.
    """
    sig = ZoneSignal(timestamp=result.timestamp, spot=result.spot)
    profile = result.l4.profile
    spot = result.spot

    if not profile:
        sig.bias = "NEUTRAL"
        sig.action = "NO DATA"
        return sig

    # ── Build Net GEX array on strike grid ──
    strikes = np.array([s.strike for s in profile])
    net_vals = np.array([s.net_gex for s in profile])  # already in M

    # ── Detect zones (same logic as visualization) ──
    # Peaks: local maxima above 70% of max positive
    max_pos = np.max(net_vals) if np.max(net_vals) > 0 else 1e9
    peak_threshold = max_pos * 0.7
    peaks = []
    for i in range(1, len(net_vals) - 1):
        if (net_vals[i] > peak_threshold
                and net_vals[i] > net_vals[i - 1]
                and net_vals[i] > net_vals[i + 1]):
            peaks.append(int(strikes[i]))

    # Troughs: local minima below 70% of min negative
    min_neg = np.min(net_vals) if np.min(net_vals) < 0 else -1e9
    trough_threshold = min_neg * 0.7
    troughs = []
    for i in range(1, len(net_vals) - 1):
        if (net_vals[i] < trough_threshold
                and net_vals[i] < net_vals[i - 1]
                and net_vals[i] < net_vals[i + 1]):
            troughs.append(int(strikes[i]))

    # Zeros: sign changes with interpolation
    zeros = []
    for i in range(len(net_vals) - 1):
        if (np.sign(net_vals[i]) != np.sign(net_vals[i + 1])
                and abs(net_vals[i]) > 0.01):
            z = strikes[i] - net_vals[i] * (
                strikes[i + 1] - strikes[i]
            ) / (net_vals[i + 1] - net_vals[i])
            zeros.append(float(z))

    sig.all_peaks = peaks
    sig.all_troughs = troughs
    sig.all_zeros = zeros

    # ── Find closest zone to spot ──
    all_zones = (
        [(p, "peak") for p in peaks]
        + [(t, "trough") for t in troughs]
        + [(z, "zero") for z in zeros]
    )
    if all_zones:
        closest = min(all_zones, key=lambda x: abs(x[0] - spot))
        sig.nearest_zone_type = closest[1]
        sig.nearest_zone_strike = closest[0]
        sig.nearest_zone_distance = abs(closest[0] - spot)

    # ── Net sentiment above/below spot ──
    lower_net = sum(s.net_gex for s in profile if s.strike < spot - STRIKE_STEP)
    upper_net = sum(s.net_gex for s in profile if s.strike > spot + STRIKE_STEP)

    sig.lower_net_positive = lower_net > 0
    sig.upper_net_negative = upper_net < 0

    # ── Derive bias ──
    # Positive below = support + Negative above = breakout upside → BULLISH
    # Negative below = breakdown + Positive above = resistance → BEARISH
    if sig.lower_net_positive and sig.upper_net_negative:
        sig.bias = "BULLISH"
    elif not sig.lower_net_positive and not sig.upper_net_negative:
        sig.bias = "BEARISH"
    elif sig.lower_net_positive and not sig.upper_net_negative:
        sig.bias = "RANGE_BOUND"  # Support below, resistance above → pinned
    else:
        sig.bias = "VOLATILE"  # Negative both sides → explosive

    # ── Action: position near closest positive-sentiment zone to spot ──
    if sig.bias == "BULLISH":
        # Closest positive-sentiment strike below spot → sell PE there
        lower_pos = sorted(
            [s for s in profile if s.strike < spot - STRIKE_STEP and s.net_gex > 0],
            key=lambda s: abs(s.strike - spot),
        )
        if lower_pos:
            zone_k = lower_pos[0].strike  # closest to spot
            sig.zone_strike = zone_k
            sig.zone_type = "sell_pe"
            sig.action = f"SELL PE near {zone_k} (support)"
        else:
            sig.action = "BULLISH but no clear support zone"
            sig.zone_type = "none"

    elif sig.bias == "BEARISH":
        # Closest positive-sentiment strike above spot → sell CE there
        upper_pos = sorted(
            [s for s in profile if s.strike > spot + STRIKE_STEP and s.net_gex > 0],
            key=lambda s: abs(s.strike - spot),
        )
        if upper_pos:
            zone_k = upper_pos[0].strike  # closest to spot
            sig.zone_strike = zone_k
            sig.zone_type = "sell_ce"
            sig.action = f"SELL CE near {zone_k} (resistance)"
        else:
            sig.action = "BEARISH but no clear resistance zone"
            sig.zone_type = "none"

    elif sig.bias == "RANGE_BOUND":
        # Support + Resistance → sell strangle at peak zones
        lower_peak = max(peaks, key=lambda p: p) if [p for p in peaks if p < spot] else None
        upper_peak = min([p for p in peaks if p > spot]) if [p for p in peaks if p > spot] else None
        if lower_peak and upper_peak:
            sig.zone_strike = lower_peak  # primary
            sig.zone_type = "sell_strangle"
            sig.action = f"SELL STRANGLE: PE at {lower_peak} / CE at {upper_peak}"
        else:
            sig.action = "RANGE BOUND — sell premium near ATM"
            sig.zone_type = "sell_strangle"

    else:  # VOLATILE
        # Negative both sides → avoid selling, consider buying
        nearest_trough = min(troughs, key=lambda t: abs(t - spot)) if troughs else None
        if nearest_trough:
            if nearest_trough > spot:
                sig.zone_strike = nearest_trough
                sig.zone_type = "buy_ce"
                sig.action = f"BUY CE near {nearest_trough} (breakout zone)"
            else:
                sig.zone_strike = nearest_trough
                sig.zone_type = "buy_pe"
                sig.action = f"BUY PE near {nearest_trough} (breakdown zone)"
        else:
            sig.action = "VOLATILE — no clear zones, stay flat"
            sig.zone_type = "none"

    return sig


def print_zone_signal(sig: ZoneSignal):
    """Print a concise zone-aligned position signal."""
    print("─" * 48)
    print("ZONE-ALIGNED POSITION SIGNAL")
    print("─" * 48)

    ts_str = sig.timestamp.strftime("%H:%M") if sig.timestamp else "N/A"
    print(f"  Time: {ts_str} | Spot: {sig.spot:,.0f}")

    if sig.all_peaks:
        print(f"  Peaks (support):  {', '.join(str(p) for p in sig.all_peaks)}")
    if sig.all_troughs:
        print(f"  Troughs (accel):  {', '.join(str(t) for t in sig.all_troughs)}")
    if sig.all_zeros:
        print(f"  Zeros (flip):     {', '.join(f'{z:.0f}' for z in sig.all_zeros)}")

    if sig.nearest_zone_strike:
        print(f"  Closest zone:     {sig.nearest_zone_type} at "
              f"{sig.nearest_zone_strike:.0f} ({sig.nearest_zone_distance:.0f} pts)")

    below_label = "✓ POSITIVE (Support)" if sig.lower_net_positive else "✗ NEGATIVE (Breakdown risk)"
    above_label = "✗ NEGATIVE (Breakout)" if sig.upper_net_negative else "✓ POSITIVE (Resistance)"
    print(f"\n  Below spot net:   {below_label}")
    print(f"  Above spot net:   {above_label}")
    print(f"  Bias:             {sig.bias}")

    print(f"\n  >>> {sig.action}")
    print("─" * 48)
