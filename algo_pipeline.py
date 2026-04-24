import pandas as pd
import numpy as np
import pytz
from datetime import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import random
import torch


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


class AlgoPipeline:
    def __init__(self):
        pass

    def calculate_features(self, name = 'btc_raw_features'):

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


        self.df['weekend'] = self.df['date'].dt.dayofweek >= 5# Add weekend indicator

        self.df.to_csv(rf"C:\Users\I'm the best\Documents\a\Algo Trading\{name}.csv", index=False)
        print(self.df)

    def fit(self, name = 'btc_raw_features', train_start = '2023-01-01', validation_start = '2025-06-01', test_start = '2026-01-01', atr_multiplyer = 3, load_encoder = False, load_model = False, sequence_length = 32, batch_size = 256):
        self.df = pd.read_csv(f'{name}.csv')
        self.df.dropna(inplace = True)
        self.df.reset_index(inplace = True, drop = True)
        self.atr_multiplyer = atr_multiplyer


        feature_cols = [c for c in list(self.df.columns) if c != 'open' and c != 'high' and c != 'low' and c != 'close' and c != 'volume' and c != 'date']

        self.df["vol_96"] = (self.df["log_return(1)"].rolling(96).std())

        self.df['label'] = triple_barrier_labels(self.df, 24, self.atr_multiplyer)
        print(self.df['label'].value_counts(True))

        df_train = self.df[ (self.df['date'] >= train_start) & (self.df['date'] < validation_start)]
        df_train_train = df_train[0:int(0.8 * len(df_train))]
        df_train_validation = df_train[int(0.8 * len(df_train)):][96:] # 1 Day embargo

        df_validation = self.df[ (self.df['date'] >= validation_start) & (self.df['date'] < test_start)]
        df_validation = df_validation[96:]# 1 day embargo 

        df_test = self.df[ (self.df['date'] >= test_start)]
        df_test = df_test[96:] # 1 day embargo between validation and test

        vol_train = self.df.loc[df_train_train.index, "vol_96"].values
        vol_train = vol_train[sequence_length - 1:]        
        vol_rank = (pd.Series(vol_train).rank(pct=True).values)

        
        stage_1_mask = vol_rank <= 0.20   # calmest 10%
        stage_2_mask = vol_rank <= 0.60   # calmest 50%
        stage_3_mask = np.ones_like(vol_rank, dtype=bool)



        scaler = StandardScaler()
        self.X_train_train = scaler.fit_transform(df_train_train[feature_cols])
        self.X_train_validation = scaler.transform(df_train_validation[feature_cols])
        self.X_validation = scaler.transform(df_validation[feature_cols])
        self.X_test = scaler.transform(df_test[feature_cols])
      

        self.y_train_train = self.df.loc[df_train_train.index, 'label']
        self.y_train_validation = self.df.loc[df_train_validation.index, 'label']
        self.y_validation = self.df.loc[df_validation.index, 'label']
        self.y_test = self.df.loc[df_test.index, 'label']
        train_train_sequence = make_dataset(self.X_train_train, self.y_train_train, seq_len= sequence_length, batch_size= batch_size)
        train_validation_sequence = make_dataset(self.X_train_validation, self.y_train_validation, shuffle = False, seq_len= sequence_length, batch_size= batch_size)
        validation_sequence = make_dataset(self.X_validation, self.y_validation, shuffle = False, seq_len= sequence_length, batch_size= batch_size)
        test_sequence = make_dataset(self.X_test, self.y_test, shuffle = False, batch_size= batch_size)

        # This triggers when the script is instructed to create its own encoder model and not to load a pre-trained one
        if load_encoder == False:

            encoder_model = build_xlstm_model(seq_len=sequence_length, n_features=self.X_train_train.shape[1])

            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-5, warmup_target=1e-3, warmup_steps=2000, decay_steps=len(train_train_sequence) * 12)
            encoder_model.compile( optimizer=tf.keras.optimizers.AdamW( learning_rate=lr_schedule,weight_decay=1e-4),loss=FocalLoss(gamma=2.0),metrics=["sparse_categorical_accuracy"])

            encoder_model.summary()
            encoder_model.fit(train_train_sequence, epochs=12, validation_data=train_validation_sequence)
            
            for layer in encoder_model.layers:
                layer.trainable = False

            # After the encoder model is trained we save it for future use
            encoder_model.save("encoder_model.keras")
        
        else:
            # Define the custom objects that the model uses to be compiled because that will be needed when we load it back up
            
            custom_objects = {"sLSTM": sLSTM, "mLSTM": mLSTM, "RMSNorm": RMSNorm, "FocalLoss": FocalLoss, "softcap": softcap,}

            encoder_model = tf.keras.models.load_model("encoder_model.keras", custom_objects=custom_objects,compile=False, safe_mode=False)  # ✅ IMPORTANT according to article

        

        
        y_val_seq = np.array([y for _, y in validation_sequence.unbatch()])

        #y_pred = np.argmax(encoder_model.predict(validation_sequence), axis=1)
        #accuracy_val, precision_val, recall_val = accuracy_score(y_val_seq, y_pred), precision_score(y_val_seq, y_pred, average = None), recall_score(y_val_seq, y_pred, average = None) 

        
        embedding_model = tf.keras.Model(inputs=encoder_model.input, outputs=encoder_model.get_layer("embedding").output) # Create an embedding model which will use the aforefitted Neural Net to extract a 128 dimensional representation of the market from its RMS layer
        
        # Extract the 128 dimensional embeddings that will be used to be feaf into the RL model
        Z_train_train = embedding_model.predict(train_train_sequence)
        Z_train_validation = embedding_model.predict(train_validation_sequence)
        Z_validation   = embedding_model.predict(validation_sequence)
        Z_test  = embedding_model.predict(test_sequence)

        
        close_train = self.df.loc[df_train_train.index, "close"].values[sequence_length - 1:]
        high_train  = self.df.loc[df_train_train.index, "high"].values[sequence_length - 1:]
        low_train   = self.df.loc[df_train_train.index, "low"].values[sequence_length - 1:]
        atr_train   = (self.df.loc[df_train_train.index, "atr_14_normalized"].values *self.df.loc[df_train_train.index, "close"].values)[sequence_length - 1:]


        model = None
        policy_kwargs = dict(net_arch=[128, 128])

        if load_model == False:
            stages = [
                ("stage_1", stage_1_mask),
                ("stage_2", stage_2_mask),
                ("stage_3", stage_3_mask),
            ]

            
            for i, (name, mask) in enumerate(stages):
                print(f"\n=== Training {name} ===")

                env = TripleBarrierTradingEnv( Z=Z_train_train[mask], close=close_train[mask], high = high_train[mask], low = low_train[mask], atr = atr_train[mask])

                if model is None:
                    model = PPO( "MlpPolicy", env, learning_rate=cosine_lr_schedule(3e-4, 1e-5), clip_range=linear_clip_schedule(0.2, 0.1), gamma=0.99, ent_coef=0.01, n_steps=2048, batch_size=256, policy_kwargs=policy_kwargs,verbose=1)
                else:
                    model.set_env(env)

                model.learn(total_timesteps=500_000)

            
            model.save("ppo_triple_barrier")

        env_train_evaluation = TripleBarrierTradingEnv(Z = Z_train_train, close= close_train, high  =high_train, low = low_train, atr= atr_train)

        close_train_val = self.df.loc[df_train_validation.index, "close"].values[sequence_length - 1:]
        high_train_val  = self.df.loc[df_train_validation.index, "high"].values[sequence_length - 1:]
        low_train_val   = self.df.loc[df_train_validation.index, "low"].values[sequence_length - 1:]
        atr_train_val   = (self.df.loc[df_train_validation.index, "atr_14_normalized"].values *self.df.loc[df_train_validation.index, "close"].values)[sequence_length - 1:]
        env_train_validation_evaluation = TripleBarrierTradingEnv( Z=Z_train_validation, close=close_train_val, high=high_train_val, low=low_train_val, atr=atr_train_val)

        close_val = self.df.loc[df_validation.index, "close"].values[sequence_length - 1:]
        high_val  = self.df.loc[df_validation.index, "high"].values[sequence_length - 1:]
        low_val   = self.df.loc[df_validation.index, "low"].values[sequence_length - 1:]
        atr_val   = (self.df.loc[df_validation.index, "atr_14_normalized"].values *self.df.loc[df_validation.index, "close"].values)[sequence_length - 1:]     
        env_validation_evaluation = TripleBarrierTradingEnv(Z = Z_validation, close = close_val, high = high_val, low = low_val, atr = atr_val)

        if load_model == True:
            model = PPO.load("ppo_triple_barrier",
                device="auto")

        model.policy.set_training_mode(False)

        # This is how you extract the results from the training set
        train_rewards, train_equity = rollout_policy(model, env_train_evaluation)
        train_sharpe = sharpe_ratio(train_rewards)

        
        print("TRAIN PERFORMANCE")
        print("-----------------")
        print(f"Total PnL     : {train_equity[-1]:.3f}")
        print(f"Sharpe Ratio  : {train_sharpe:.3f}")
        print(f"Steps         : {len(train_rewards)}")


        plt.figure(figsize=(12, 4))
        plt.plot(train_equity)
        plt.title("Training Equity Curve")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.show()


        # This is how to extract the results for the train_validaiton set
        train_validation_rewards, train_validation_equity = rollout_policy(model, env_train_validation_evaluation)
        train_validation_sharpe = sharpe_ratio(train_validation_rewards)

        
        print("\n TRAIN VALIDATION PERFORMANCE")
        print("--------------------")
        print(f"Total PnL     : {train_validation_equity[-1]:.3f}")
        print(f"Sharpe Ratio  : {train_validation_sharpe:.3f}")
        print(f"Steps         : {len(train_validation_rewards)}")

        plt.figure(figsize=(12, 4))
        plt.plot(train_validation_equity)
        plt.title("Train Validation Equity Curve")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.show()



        # This is how to extract the results for the validation set
        validation_rewards, validation_equity = rollout_policy(model, env_validation_evaluation)
        validation_sharpe = sharpe_ratio(validation_rewards)

        
        print("\n VALIDATION PERFORMANCE")
        print("--------------------")
        print(f"Total PnL     : {validation_equity[-1]:.3f}")
        print(f"Sharpe Ratio  : {validation_sharpe:.3f}")
        print(f"Steps         : {len(validation_rewards)}")

        plt.figure(figsize=(12, 4))
        plt.plot(validation_equity)
        plt.title("Train Validation Equity Curve")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.show()


        
        trade_rewards_train = train_rewards[np.abs(train_rewards) >= 1.0]
        trade_rewards_train_validation   = train_validation_rewards[np.abs(train_validation_rewards) >= 1.0]
        trade_rewards_validation = validation_rewards[np.abs(validation_rewards) >= 1.0]

        print("\nTRAIN TRADES")
        print("-----------")
        print("Trades  :", len(trade_rewards_train))
        print("Win rate:", np.mean(trade_rewards_train > 0))

        print("\nTRAIN VAL TRADES")
        print("---------")
        print("Trades  :", len(trade_rewards_train_validation))
        print("Win rate:", np.mean(trade_rewards_train_validation > 0))


        print("\nVAL TRADES")
        print("---------")
        print("Trades  :", len(trade_rewards_validation))
        print("Win rate:", np.mean(trade_rewards_validation > 0))



def triple_barrier_labels(df: pd.DataFrame,horizon: int = 24,atr_mult: float = 1.5,atr_col: str = "atr_14_normalized",close_col: str = "close") -> pd.Series:
    """
    Triple Barrier labeling (causal, per-bar).

    Returns:
        Series with labels in {-1, 0, +1}
    """
    close = df[close_col].values
    atr   = (df[atr_col] * close).values  # de-normalize ATR
    high  = df["high"].values
    low   = df["low"].values

    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        if i + 1 >= len(df):
            continue

        upper = close[i] + atr_mult * atr[i]
        lower = close[i] - atr_mult * atr[i]

        end = min(i + horizon + 1, len(df))

        for j in range(i + 1, end):
            if high[j] >= upper:
                labels[i] = 1
                break
            if low[j] <= lower:
                labels[i] = -1
                break
        
    labels = labels + 1
        # else: stays 0 (expiry)

    return labels


def make_dataset(X, y, seq_len=32, batch_size=64, shuffle=True):
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i:i + seq_len])
        ys.append(y.iloc[i + seq_len - 1])

    Xs = np.asarray(Xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((Xs, ys))
    if shuffle:
        ds = ds.shuffle(20_000)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@tf.function
def softcap(x, a=10.0):
    return a * tf.tanh(x / a)


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.scale = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True
        )

    def call(self, x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / rms * self.scale




class sLSTM(tf.keras.layers.Layer):
    def __init__(self, dim, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.linear = tf.keras.layers.Dense(4 * dim, kernel_initializer="orthogonal")
        self.norm = RMSNorm(dim)

        self.input_bias = self.add_weight(
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
            name="input_bias"
        )

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        c = tf.zeros((B, self.dim))
        n = tf.zeros((B, self.dim))
        m = tf.zeros((B, self.dim))

        outputs = tf.TensorArray(dtype=tf.float32, size=T)

        def step(t, c, n, m, outputs):
            xt = x[:, t, :]

            i_pre, f_pre, o_pre, z = tf.split(self.linear(xt), 4, axis=-1)

            i = softcap(i_pre + self.input_bias)
            f = -tf.nn.softplus(f_pre)

            m_next = tf.maximum(f + m, i)
            i_scaled = tf.exp(i - m_next)
            f_scaled = tf.exp(f + m - m_next)

            c = f_scaled * c + i_scaled * tf.tanh(z)
            n = f_scaled * n + i_scaled

            h = tf.nn.sigmoid(o_pre) * (c / (n + 1e-6))
            outputs = outputs.write(t, h)

            return t + 1, c, n, m_next, outputs

        t0 = tf.constant(0)
        _, _, _, _, outputs = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=step,
            loop_vars=(t0, c, n, m, outputs),
            parallel_iterations=1
        )

        out = tf.transpose(outputs.stack(), perm=[1, 0, 2])
        out = self.norm(out)
        return self.dropout(out, training=training)





class mLSTM(tf.keras.layers.Layer):
    def __init__(self, dim, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.q_proj = tf.keras.layers.Dense(dim)
        self.k_proj = tf.keras.layers.Dense(dim)
        self.v_proj = tf.keras.layers.Dense(dim)

        self.gate_proj = tf.keras.layers.Dense(3 * dim)
        self.norm = RMSNorm(dim)

        self.input_bias = self.add_weight(
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True,
            name="input_bias"
        )

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = self.dim

        M = tf.zeros((B, D, D))
        n_vec = tf.zeros((B, D))
        m_state = tf.zeros((B, D))

        outputs = tf.TensorArray(tf.float32, size=T)

        def step(t, M, n_vec, m_state, outputs):
            xt = x[:, t, :]

            q = self.q_proj(xt)
            k = self.k_proj(xt) / tf.sqrt(tf.cast(D, tf.float32))
            v = self.v_proj(xt)

            i_pre, f_pre, o_pre = tf.split(self.gate_proj(xt), 3, axis=-1)

            log_i = softcap(i_pre + self.input_bias)
            log_f = -tf.nn.softplus(f_pre)

            m_next = tf.maximum(log_f + m_state, log_i)

            i_scaled = tf.exp(log_i - m_next)
            f_scaled = tf.exp(log_f + m_state - m_next)

            vkT = tf.einsum("bi,bj->bij", v, k)
            M = f_scaled[:, :, None] * M + i_scaled[:, :, None] * vkT
            n_vec = f_scaled * n_vec + i_scaled * k

            h_raw = tf.einsum("bij,bj->bi", M, q)
            denom = tf.maximum(
                tf.abs(tf.einsum("bi,bi->b", n_vec, q))[:, None],
                1e-6
            )

            h = tf.nn.sigmoid(o_pre) * (h_raw / denom)
            outputs = outputs.write(t, h)

            return t + 1, M, n_vec, m_next, outputs

        t0 = tf.constant(0)
        _, M, n_vec, m_state, outputs = tf.while_loop(
            cond=lambda t, *_: t < T,
            body=step,
            loop_vars=(t0, M, n_vec, m_state, outputs),
            parallel_iterations=1
        )

        out = tf.transpose(outputs.stack(), perm=[1, 0, 2])
        out = self.norm(out)
        return self.dropout(out, training=training)




def build_xlstm_model(seq_len=64, n_features=25, dim=128):
    inputs = tf.keras.Input(shape=(seq_len, n_features))

    # Initial projection to model dimension
    x = tf.keras.layers.Dense(dim)(inputs)

    # --- Block 1: sLSTM (Scalar Memory) ---
    # We use Pre-Norm: Norm -> Layer -> Add
    res = x
    x = tf.keras.layers.LayerNormalization()(x)
    x = sLSTM(dim, dropout=0.2)(x)
    x = tf.keras.layers.Add()([x, res])

    # --- Block 2: mLSTM (Matrix Memory) ---
    res = x
    x = tf.keras.layers.LayerNormalization()(x)
    x = mLSTM(dim, dropout=0.2)(x)
    x = tf.keras.layers.Add()([x, res])

    # --- Feature Extraction ---
    # Instead of Average Pooling, we take the last state (the "Now" bar)
    # This is more reactive to immediate market triggers.
    x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)

    # Optional: A small dense bottleneck for higher reasoning
    x = tf.keras.layers.Dense(dim // 2, activation='gelu', name="embedding")(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output: 3 classes (0: Down, 1: Flat, 2: Up)
    outputs = tf.keras.layers.Dense(3)(x) 

    return tf.keras.Model(inputs, outputs)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=(0.3, 0.4, 0.3)):
        super().__init__()
        self.gamma = gamma
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def call(self, y_true, logits):
        #logits = tf.clip_by_value(logits, -20.0, 20.0)
        # Ensure integer labels
        y_true = tf.cast(y_true, tf.int32)

        # One-hot
        y_true_oh = tf.one_hot(y_true, depth=3)

        # ✅ LOG-SOFTMAX (stable)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.exp(log_probs)

        # Cross-entropy
        ce = -y_true_oh * log_probs

        # Focal scaling
        focal = self.alpha * tf.pow(1.0 - probs, self.gamma) * ce

        return tf.reduce_sum(focal, axis=-1)






class TripleBarrierTradingEnv(gym.Env):
    def __init__(self, Z, close, high, low, atr, max_holding=20, spread_cost=0.05, tp_sl_mult = 4.0):
        super().__init__()

        self.Z = Z.astype(np.float32)
        self.close = close
        self.high = high
        self.low = low
        self.atr = atr

        self.tp_sl_mult = tp_sl_mult

        self.max_holding = max_holding
        self.spread_cost = spread_cost

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(Z.shape[1],),dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.position = 0
        self.entry_price = None
        self.tp = None
        self.sl = None
        self.holding_time = 0

        obs = self.Z[self.t]
        info = {}

        return obs, info

    def step(self, action):
        reward = 0.0

        # ENTRY LOGIC
        if self.position == 0:
            if action == 1:
                self.position = 1
            elif action == 2:
                self.position = -1

            if self.position != 0:
                self.entry_price = self.close[self.t]
                atr = self.atr[self.t]

                if self.position == 1:
                    self.tp = self.entry_price + self.tp_sl_mult * atr
                    self.sl = self.entry_price - 1.0 * atr
                else:
                    self.tp = self.entry_price - self.tp_sl_mult * atr
                    self.sl = self.entry_price + 1.0 * atr

                reward -= self.spread_cost
                self.holding_time = 0

        # POSITION MANAGEMENT
        else:
            self.holding_time += 1

            if self.position == 1:
                if self.high[self.t] >= self.tp:
                    reward += self.tp_sl_mult
                    self.position = 0
                elif self.low[self.t] <= self.sl:
                    reward -= 1.0
                    self.position = 0

            elif self.position == -1:
                if self.low[self.t] <= self.tp:
                    reward += self.tp_sl_mult
                    self.position = 0
                elif self.high[self.t] >= self.sl:
                    reward -= 1.0
                    self.position = 0

            # TIME EXPIRY
            if self.position != 0 and self.holding_time >= self.max_holding:
                pnl = (self.close[self.t] - self.entry_price) / self.atr[self.t]
                reward += pnl * self.position
                self.position = 0

        self.t += 1

        terminated = self.t >= len(self.Z) - 1
        truncated = False

        obs = self.Z[self.t]
        info = {}

        return obs, reward, terminated, truncated, info


def cosine_lr_schedule(initial_lr, final_lr):
    def schedule(progress):
        return final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * progress))
    return schedule

def linear_clip_schedule(start, end):
    def schedule(progress):
        return end + (start - end) * progress
    return schedule



def rollout_policy(model, env):
    """
    Roll out a trained SB3 policy on a Gymnasium environment.
    Returns per-step rewards and the equity curve.
    """

    rewards = []
    equity_curve = []

    obs, info = env.reset(seed = SEED)
    terminated = False
    truncated = False

    cumulative_pnl = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        cumulative_pnl += reward
        equity_curve.append(cumulative_pnl)

    return np.array(rewards), np.array(equity_curve)


def sharpe_ratio(returns, eps=1e-8):
    mean_ret = returns.mean()
    std_ret = returns.std() + eps
    ann_factor = np.sqrt(96 * 365)
    return mean_ret / std_ret * ann_factor



if __name__ == "__main__":
    pipeline = AlgoPipeline()
    #pipeline.calculate_features()
    pipeline.fit(load_encoder= True, load_model= False,sequence_length=32, batch_size = 256, validation_start= '2025-06-01', test_start = '2026-01-01')