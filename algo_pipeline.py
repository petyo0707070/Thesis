import pandas as pd
import numpy as np
import pandas_ta as ta
import pytz
from datetime import time

class AlgoPipeline:
    def __init__(self):
        self.df = pd.read_excel(r"C:\Users\I'm the best\Documents\a\Algo Trading\btc_ohlc_15.xlsx")# Read data
        self.df['ticks'] = pd.to_datetime(self.df['ticks']).dt.tz_localize(None).dt.tz_localize('UTC')# Convert ticks to datetime
        self.df.rename(columns={'ticks': 'date'}, inplace=True)# Rename ticks to datetime
        self.df['Date'] = self.df['date'].values
        self.df.set_index('date', inplace=True)
        self.df.rename(columns = {'Date': 'date'}, inplace=True)

        if self.df['date'].dt.tz is None:
            self.df['date'] = self.df['date'].dt.tz_localize('UTC')
        else:
            self.df['date'] = self.df['date'].dt.tz_convert('UTC')

    def calculate_features(self, name = 'btc_raw_features'):
        self.df['log_return(1)'] = np.log(self.df['close'] / self.df['close'].shift(1))# Calculate log return

        # Volatility features
        self.df['atr_14_normalized'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=14) / self.df['close']
        self.df['atr_5_normalized'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=5) / self.df['close']
        self.df['rolling_vol_20'] = self.df['log_return(1)'].rolling(window=20).std()# Calculate rolling volatility
        self.df['volatility_skew'] = ( self.df['high'] - np.maximum(self.df['open'], self.df['close']) - (np.minimum(self.df['open'], self.df['close']) - self.df['low']) ) / self.df['close']# Calculate volatility skew

        # Trend / Mean Reversion
        self.df['ema_20_distance'] = (self.df['close'] -  ta.ema(self.df['close'], length=20) ) / ta.ema(self.df['close'], length=20)
        self.df['ema_50_distance'] = (self.df['close'] -  ta.ema(self.df['close'], length=50) ) / ta.ema(self.df['close'], length=50)
        self.df['ema_200_distance'] = (self.df['close'] -  ta.ema(self.df['close'], length=200) ) / ta.ema(self.df['close'], length=200)
        self.df['ema_20/ema_50'] = ta.ema(self.df['close'], length=20) / ta.ema(self.df['close'], length=50)
        self.df['ema_50/ema_200'] = ta.ema(self.df['close'], length=50) / ta.ema(self.df['close'], length=200)

        # Momentum
        tema_macd = ta.macd(self.df['close'], fast=12, slow=26, signal=9, mamode="tema")
        self.df['tema_macd'] = tema_macd['MACD_12_26_9']
        self.df['tema_signal'] = tema_macd['MACDs_12_26_9']
        self.df['tema_hist'] = tema_macd['MACDh_12_26_9']

        self.df['rsi(14)'] = ta.rsi(self.df['close'])

        bbands = self.df.ta.bbands(length=20, std=2)
        self.df['bb_bandwidth'] = bbands['BBB_20_2.0']
        self.df['bb_percent_b'] = bbands['BBP_20_2.0']

        # Log Returns at different lookbacks
        self.df['log_return(5)'] = np.log(self.df['close'] / self.df['close'].shift(5))# Calculate log return
        self.df['log_return(15)'] = np.log(self.df['close'] / self.df['close'].shift(15))# Calculate log return
        self.df['log_return(60)'] = np.log(self.df['close'] / self.df['close'].shift(60))# Calculate log return

        # Helper Functions that will be used to compute SMT divergence and daily swings

        def _resample_to_daily(df):
    # If the index isn't a datetime, try to use a column
            if not isinstance(df.index, pd.DatetimeIndex):
                # Check if 'timestamp' is a column and set it
                if 'date' in df.columns:
                    df = df.set_index('date')
                else:
                    raise ValueError("DataFrame index must be DatetimeIndex for resampling.")

            return (df.resample("1D").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),   close=("close", "last"), volume=("volume", "sum")).dropna(subset=["close"]))

        def _causal_swing_prices(high: pd.Series, low: pd.Series, window: int = 3):
            """
            Swing high/low prices with zero lookahead.

            center=True rolling requires `window` future bars to compute the local
            max/min around bar i.  We cancel that lookahead with .shift(window):
            the value produced at bar i+window tells us what happened at bar i —
            by which point all `window` right-side confirmation bars have closed.

                bar:       i-w  ...  i  ...  i+w
                rolling sees:  [i-w  ...  i  ...  i+w]   ← lookahead
                .shift(w) moves result to bar i+w         ← no lookahead
            """
            sh = high.where(high == high.rolling(2 * window + 1, center=True).max()).shift(window)
            sl = low.where( low  == low.rolling(2 * window + 1,  center=True).min()).shift(window)
            return sh, sl

        def _prev_swing(swing: pd.Series) -> pd.Series:
            """
            Returns the swing price that preceded the most-recent confirmed swing.

            Walk-through with swing confirmations at bars 10, 20, 35 (prices 100, 110, 105):
            ffill           :  [NaN…100…100…110…110…105…]
            ffill.shift(1)  :  [NaN…NaN…100…100…110…110…]   ← value just before each new swing
            .where(new_swing):  NaN   NaN  100   NaN  110   NaN
            .ffill()         :  NaN   NaN  100   100  110   110  ← carried forward = "previous swing"

            All operations are purely backward-looking. ✓
            """
            ffilled       = swing.ffill()
            prev_at_new   = ffilled.shift(1).where(swing.notna())
            return prev_at_new.ffill()

        def _broadcast_daily_to_15min(daily: pd.Series, bar_index: pd.DatetimeIndex) -> pd.Series:
            """
            Map a daily Series onto 15-min bars by date.

            daily.shift(1) is applied before mapping so that bars forming
            on day D only see day D-1's completed value — not today's partial bar.
            """
            lagged = daily.shift(1)
            lagged.index = lagged.index.normalize()          # strip intraday time component
            bar_dates = bar_index.normalize()
            return lagged.reindex(bar_dates).set_axis(bar_index)

        def compute_smt_divergence(gold_15min:   pd.DataFrame, dxy_15min:    pd.DataFrame, swing_window: int = 3,) -> pd.Series:
            """
            DXY-Gold Smart Money Theory divergence, mapped to 15-min bars.

            Signal
            ------
            +1  Bullish SMT : Gold AND DXY both print a lower swing low.
                            They should diverge (DXY up → Gold down), so both dropping
                            together means Gold's sell-off is suspect — potential long.
            -1  Bearish SMT : Gold AND DXY both print a higher swing high.
                            They should diverge (DXY down → Gold up), so both rallying
                            together means Gold's rally is suspect — potential short.
            0  No divergence.

            No-lookahead guarantees
            -----------------------
            • Swing confirmed only after `swing_window` daily bars have elapsed (shift).
            • Today's daily bar is excluded until it closes (shift(1) before broadcast).
            • Previous-swing reference uses only prior confirmed swings (ffill + shift(1)).

            Parameters
            ----------
            gold_15min   : 15-min OHLCV DataFrame for XAU/USD  (DatetimeIndex)
            dxy_15min    : 15-min OHLCV DataFrame for DXY       (DatetimeIndex)
            swing_window : bars each side used to confirm a daily swing point
            """
            gold_d = _resample_to_daily(gold_15min)
            dxy_d  = _resample_to_daily(dxy_15min)

            # Causal swing detection (delayed by swing_window daily bars)
            gold_sh, gold_sl = _causal_swing_prices(gold_d["high"], gold_d["low"], swing_window)
            dxy_sh,  dxy_sl  = _causal_swing_prices(dxy_d["high"],  dxy_d["low"],  swing_window)

            # Most-recent swing and the one before it
            gold_sl_cur, gold_sl_prev = gold_sl.ffill(), _prev_swing(gold_sl)
            gold_sh_cur, gold_sh_prev = gold_sh.ffill(), _prev_swing(gold_sh)
            dxy_sl_cur,  dxy_sl_prev  = dxy_sl.ffill(),  _prev_swing(dxy_sl)
            dxy_sh_cur,  dxy_sh_prev  = dxy_sh.ffill(),  _prev_swing(dxy_sh)

            bullish_smt = (gold_sl_cur < gold_sl_prev) & (dxy_sl_cur < dxy_sl_prev)
            bearish_smt = (gold_sh_cur > gold_sh_prev) & (dxy_sh_cur > dxy_sh_prev)

            daily_smt = pd.Series(0, index=gold_d.index, dtype=int)
            daily_smt[bullish_smt] =  1
            daily_smt[bearish_smt] = -1

            # Broadcast to 15-min (shift(1) inside _broadcast_daily_to_15min)
            return (_broadcast_daily_to_15min(daily_smt, gold_15min.index).fillna(0).astype(int).rename("smt_divergence"))

        def compute_daily_swing_levels(gold_15min:   pd.DataFrame,swing_window: int = 3,) -> pd.DataFrame:
            """
            Appends daily swing high/low levels (and derived distances) to gold_15min.

            Added columns
            -------------
            daily_swing_high   : price of the most-recent confirmed daily swing high
            daily_swing_low    : price of the most-recent confirmed daily swing low
            dist_to_swing_high : (swing_high − mid) / mid  — positive means price is below
            dist_to_swing_low  : (mid − swing_low)  / mid  — positive means price is above

            No-lookahead guarantees
            -----------------------
            Same as compute_smt_divergence: causal swing detection + daily shift(1).
            """
            gold_d = _resample_to_daily(gold_15min)
            sh, sl = _causal_swing_prices(gold_d["high"], gold_d["low"], swing_window)

            sh_daily = sh.ffill()   # carry forward until a new swing is confirmed
            sl_daily = sl.ffill()

            df = gold_15min.copy()
            df["daily_swing_high"] = _broadcast_daily_to_15min(sh_daily, gold_15min.index)
            df["daily_swing_low"]  = _broadcast_daily_to_15min(sl_daily, gold_15min.index)

            mid = (df["high"] + df["low"]) / 2
            df["dist_to_swing_high"] = (df["daily_swing_high"] - mid) / mid
            df["dist_to_swing_low"]  = (mid - df["daily_swing_low"])  / mid

            return df[["dist_to_swing_high", "dist_to_swing_low"]]

        # Swing levels
        self.df[["dist_to_swing_high", "dist_to_swing_low"]] = compute_daily_swing_levels(self.df)

        def session_overlaps():
            df = self.df.copy()
            # Time variables
            def is_open(row_utc, timezone_str, open_time, close_time):
                # Convert UTC time to local market time
                local_time = row_utc.astimezone(pytz.timezone(timezone_str)).time()
                return open_time <= local_time <= close_time
            
            markets = {'London': {'tz': 'Europe/London', 'open': time(7, 0), 'close': time(17, 00)}, 'NewYork': {'tz': 'America/New_York', 'open': time(8, 00), 'close': time(17, 0)},'Tokyo': {'tz': 'Asia/Tokyo', 'open': time(8, 0), 'close': time(19, 0)}}
            for m, config in markets.items():
                df[f'{m}_Open'] = df['date'].apply(lambda x: is_open(x, config['tz'], config['open'], config['close']))

            is_weekday = df['date'].dt.dayofweek < 5

            df['tokyo_london'] = (df['Tokyo_Open'] & df['London_Open']) & is_weekday
            df['london_ny'] = (df['London_Open'] & df['NewYork_Open']) & is_weekday
            df['tokyo_ny'] = (df['NewYork_Open'] & df['Tokyo_Open']) & is_weekday

            return df[['tokyo_ny', 'london_ny', 'tokyo_london']]

        self.df['weekend'] = self.df['date'].dt.dayofweek >= 5# Add weekend indicator
        self.df[['tokyo_ny', 'london_ny', 'tokyo_london']] = session_overlaps()# Add session overlap indicators

        self.df.to_csv(rf"C:\Users\I'm the best\Documents\a\Algo Trading\{name}.csv", index=False)
        print(self.df)

if __name__ == "__main__":
    pipeline = AlgoPipeline()
    pipeline.calculate_features()