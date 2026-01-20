import pandas as pd
import os
import datetime as dt
import sys
import numpy as np



class PartionOptionData():
    def __init__(self, filepath = r'D:\Option Data\options_1.csv'):

        output_dir = r'D:\Option Data\stock_options'
        chunk_size = 500_000 # We will read th file chunk by chunk and save into the different csv_s dynamically


        for chunk in pd.read_csv(filepath, chunksize=chunk_size): # This Loops over each chunk

            chunk['strike_price'] = chunk['strike_price'] / 1000
            chunk.drop(['index_flag', 'issuer', 'exercise_style'], axis = 'columns', inplace = True)
            print(chunk)


            for ticker, data in chunk.groupby('ticker'): # This loops over each ticker in the chunk
                ticker_file = os.path.join(output_dir, f'{ticker}.csv')

                if not os.path.isfile(ticker_file):
                    data.to_csv(ticker_file, index=False)
                else:
                    data.to_csv(ticker_file, index=False, header=False, mode='a')


class PreprocessOptionData():

    def __init__(self, ticker_number = 1):

        def filter_valid_events(df): # Basically insure that we are only working with full events not some which are cut off
            def is_valid(group):

                has_min_rows = group['t'].nunique() >= 20

                t0_rows = group[group['t'] == 0]

                has_t_zero = (group['t'] == 0).any()
                if not has_t_zero:
                    return False
                
                has_t_one = (group['t'] == 1).any()

                dte_at_t0 = (t0_rows['DTE'] == 0).any()


                t_logic_valid = has_t_one or (not has_t_one and dte_at_t0)
                return has_min_rows and t_logic_valid
            
            df_filtered = df.groupby('event_identifier').filter(is_valid)
            return df_filtered


        def filter_events_fast(df):
            # 1. Get unique 't' counts per group (Vectorized)
            # transform('nunique') maps the result back to every row index
            unique_t_counts = df.groupby('event_identifier')['t'].transform('nunique')

            # 2. Check for existence of t=0 and t=1 per group
            # We create boolean series and use transform('any')
            has_t0 = df['t'].eq(0).groupby(df['event_identifier']).transform('any')
            has_t1 = df['t'].eq(1).groupby(df['event_identifier']).transform('any')

            # 3. Check for DTE=0 specifically when t=0
            # We mark rows that satisfy the condition, then spread that "True" to the whole group
            is_dte0_at_t0 = (df['t'] == 0) & (df['DTE'] == 0)
            group_has_dte0_at_t0 = is_dte0_at_t0.groupby(df['event_identifier']).transform('any')

            # 4. Combine all logic into one final mask
            # Logic: (Min 20 days) AND (Has T0) AND (Has T1 OR Expired at T0)
            mask = (unique_t_counts >= 20) & \
                (has_t0) & \
                (has_t1 | group_has_dte0_at_t0)

            return df[mask]

        def filter_events(df):
            required_t = {0, -15}
            valid_mask = df.groupby('event_identifier')['t'].transform(lambda x: required_t.issubset(set(x)))
            df = df[valid_mask].copy()

            counts = df.groupby('event_identifier')['t'].nunique()
            print(counts)
            valid_events = counts[counts >= 15].index
            filtered_df = df[df['event_identifier'].isin(valid_events)]

            return filtered_df

        
        with open(rf"C:\Users\I'm the best\Documents\a\Earnings Estimation\Thesis\tickers_{ticker_number}.txt", 'r') as f:
            ticker_list = f.read().splitlines()
        
        self.load_earnings_data()

        itt = 1
        
        for t in ticker_list:
            print(f'Itteration number {itt}, ticker {t}')
            itt += 1

            t_earnings = self.earnings_df[self.earnings_df['OFTIC'] == t].copy()
            t_earnings = t_earnings.sort_values('ANNDATS')
            df = pd.read_csv(fr'D:\Option Data\stock_options\{t}.csv', engine = 'python') # These are all the options chains over all the days for the given ticker
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
            df['exdate'] = pd.to_datetime(df['exdate']).dt.normalize()
            df = df.sort_values('date') # Sort by date because for some reason it is not always te case

            df = pd.merge_asof(df,t_earnings[['ANNDATS', 'Event Type']],left_on='date',right_on = 'ANNDATS', direction = 'nearest') # Merge the date with the nearest earnings event
            df['t_adjustment'] = df['Event Type'].apply(lambda x: 1 if x == 'Before Market' else 0) # Basically we need to move the t up by 1 if the event is Before Market
            df['t'] = np.busday_count(df['ANNDATS'].values.astype('datetime64[D]'), df['date'].values.astype('datetime64[D]'))
            df['t'] = df['t'] + df['t_adjustment']
            df['day_week'] = df['ANNDATS'].dt.dayofweek
            df['After Market'] = df['Event Type'].apply(lambda x: 1 if x == 'After Market' else 0)
            df['event_identifier'] = df.groupby('ANNDATS', sort=True).ngroup() # This basically simplifies my work later on as it assigns each row to a given earnings event
            df = df[df['exdate'] >= df['ANNDATS']] # Keep only the option chains which expire on or after the event, those are really those we care about
            df = df[(df['t'] >= -15) & (df['t'] <= 1)] # Keep only the options in the event windows
            df.reset_index(drop = True, inplace = True)
            df.drop(['t_adjustment', 'ANNDATS', 'Event Type'], axis = 'columns', inplace = True)
            df['DTE'] = (df['exdate'] - df['date']).dt.days # Use this to get Days Till Expiration (DTE)
            df = df[df['date'] <= '2024-12-31'] # Get only data from 01.01.2014 - 31.12.2024
            df = filter_events(df) # Remove Uncomplete earnings windows

            df.to_csv(rf"D:\Option Data\stock_options\{t}.csv", index = False)

    def load_earnings_data(self):
        self.earnings_df = pd.read_csv(r'D:\Option Data\earnings_dates.csv') # Load the earnings df

        # Define the start and end of trading days i.e. no announcement during them are to be considered
        start = dt.time(9, 30)
        end = dt.time(16, 0) 

        self.earnings_df["ANNTIMS"] = pd.to_datetime(self.earnings_df["ANNTIMS"], format="%H:%M:%S").dt.time
        self.earnings_df['ANNDATS'] = pd.to_datetime(self.earnings_df['ANNDATS']).dt.normalize()
        self.earnings_df = self.earnings_df[~self.earnings_df["ANNTIMS"].between(start, end)] # We do not want any event that happens during regular trading hours
        self.earnings_df['Event Type'] = self.earnings_df['ANNTIMS'].apply(lambda x: 'Before Market' if x < start else 'After Market') # Define whether the event happen Before or After Market
        self.earnings_df.drop(['PDICITY', 'TICKER'], axis = 1, inplace = True) # 


class CalculateFeatureMatrixes():

    def __init__(self, ticker_number = 1, verbose = False):

        self.verbose = verbose

        self.stocks_df = pd.read_csv('D:\Option Data\stocks.csv')
        self.stocks_df['DlyCalDt'] = pd.to_datetime(self.stocks_df['DlyCalDt']).dt.normalize()
        self.stocks_df['Ticker'] = self.stocks_df['Ticker'].astype(str).str.strip().str.upper() # Standardize the string name to ensure correct merges
        self.stocks_df = self.stocks_df.drop_duplicates(subset=['DlyCalDt', 'Ticker'], keep='first')
        self.stock_tickers = self.stocks_df['Ticker'].unique().tolist()
        print(self.stocks_df)

        self.vix = pd.read_csv(r'D:\Option Data\vix.csv')
        self.vix['Date'] = pd.to_datetime(self.vix['Date'])
        self.vix = self.vix.set_index('Date')

        self.sp500 = pd.read_csv(r'D:\Option Data\sp500.csv')
        self.sp500['DlyCalDt'] = pd.to_datetime(self.sp500['DlyCalDt'])
        self.sp500['Return'] = (self.sp500['spindx'] - self.sp500['spindx'].shift(1)) / self.sp500['spindx'].shift(1)

        self.risk_free = pd.read_csv(r'D:\Option Data\riskfree.csv')
        self.risk_free = self.risk_free[self.risk_free['TTERMLBL'] == 'CRSP Risk Free - 4 week (Nominal)']
        self.risk_free['CALDT'] = pd.to_datetime(self.risk_free['CALDT']).dt.normalize()
        self.risk_free['TDBIDYLD'] = (1 + self.risk_free['TDBIDYLD']) ** 365
        self.risk_free['TDYLD'] = (1 + self.risk_free['TDYLD']) ** 365
        self.risk_free['RF'] = (self.risk_free['TDBIDYLD'] + self.risk_free['TDYLD']) / 2 - 1
        self.risk_free = self.risk_free[['CALDT', 'RF']]
        self.risk_free = self.risk_free.set_index('CALDT')


        # Get all the tickers in the current batch
        with open(rf"C:\Users\I'm the best\Documents\a\Earnings Estimation\Thesis\tickers_{ticker_number}.txt", 'r') as f:
            self.ticker_list = f.read().splitlines()


        #self.ticker_list = self.ticker_list[33:]
        self.calculate_features()

    def calculate_features(self): # Calculate the first feature matrix that would be fead into the model
        self.abnormal_volume_call = pd.read_csv(r'D:\Option Data\feature_matrixes\abnormal_volume_call.csv', index_col =0)
        self.abnormal_volume_put = pd.read_csv(r'D:\Option Data\feature_matrixes\abnormal_volume_put.csv', index_col =0)
        self.after_market = pd.read_csv(r'D:\Option Data\feature_matrixes\after_market.csv', index_col =0)
        self.atm_call_open_interest_ratio = pd.read_csv(r'D:\Option Data\feature_matrixes\atm_call_open_interest_ratio.csv', index_col =0)
        self.atm_put_open_interest_ratio = pd.read_csv(r'D:\Option Data\feature_matrixes\atm_put_open_interest_ratio.csv', index_col = 0)
        self.beta = pd.read_csv(r'D:\Option Data\feature_matrixes\beta.csv', index_col = 0)
        self.call_put_ratio = pd.read_csv(r'D:\Option Data\feature_matrixes\call_put_ratio.csv', index_col = 0)
        self.day_of_week = pd.read_csv(r'D:\Option Data\feature_matrixes\day_of_week.csv', index_col = 0)
        self.implied_kurt_change = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_kurt_change.csv', index_col = 0)
        self.implied_skew_change = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_skew_change.csv', index_col = 0)
        self.implied_move = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_move.csv', index_col = 0)
        self.implied_vol = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_vol.csv', index_col = 0)
        self.iv_slope = pd.read_csv(r'D:\Option Data\feature_matrixes\iv_slope.csv', index_col = 0)
        self.log_market_cap = pd.read_csv(r'D:\Option Data\feature_matrixes\log_market_cap.csv', index_col = 0)
        self.realized_move_pct = pd.read_csv(r'D:\Option Data\feature_matrixes\realized_move_pct.csv', index_col = 0)
        self.realized_move_pct_abs = pd.read_csv(r'D:\Option Data\feature_matrixes\realized_move_pct_abs.csv', index_col = 0)
        self.vix_ = pd.read_csv(r'D:\Option Data\feature_matrixes\vix.csv', index_col = 0)
        self.date = pd.read_csv(r'D:\Option Data\feature_matrixes\date.csv', index_col = 0, dtype=object)




        self.pnl_bid_ask_df = pd.read_csv(r'D:\Option Data\feature_matrixes\pnl_bid_ask_backtest.csv', index_col = 0)
        self.pnl_realistic_df = pd.read_csv(r'D:\Option Data\feature_matrixes\pnl_realistic_backtest.csv', index_col = 0)

        itt = 0
        for t in self.ticker_list:

            if t not in self.stock_tickers: # If the ticker is not present in the stocks OHLC from CRSPR than set all the features to NaN or whatever
                continue


            print(f'At ticker {t}, itteration {itt}')
            itt += 1
            df = pd.read_csv(rf'D:\Option Data\stock_options\{t}.csv', engine = 'python')
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
            df['exdate'] = pd.to_datetime(df['exdate']).dt.normalize()
            df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
            df.drop_duplicates(inplace = True, keep = 'first', subset = ['optionid', 'date']) # Apparently some options have duplicate rows with differe t optionid so we need to drop also for date, exdate, cpflag and strike_price
            df.drop_duplicates(inplace = True, keep = 'first', subset = ['date', 'exdate', 'cp_flag', 'strike_price'])
            df = df[df['date'] <= '2024-12-31']

            stock_ticker_df = self.stocks_df[self.stocks_df['Ticker'] == t]

            # Add the close prices to the option df
            df = df.merge(self.stocks_df[['DlyCalDt', 'Ticker', 'DlyClose', 'DlyCap']], left_on=['date', 'ticker'], right_on=['DlyCalDt', 'Ticker'], how='left')
            
            events = df['event_identifier'].unique().tolist()

            for event in events:
                df_ = df[df['event_identifier'] == event]

                # Get all the option chains at t = 0 and t = 1
                options_t0 = df_[df_['t'] == 0].copy()
                options_t1 = df_[df_['t'] == 1].copy()

                # Get the date of t = 0, we will use it to extract marketcap and vix
                date_t0 = options_t0['date'].values[0]

                date_t0_ts = pd.Timestamp(date_t0)# Convert the date into a timestring which would be used to extract the quarte and year in order to save into the relevant feature matrix
                q_string = f"{date_t0_ts.year}-Q{date_t0_ts.quarter}"


                if options_t0['DlyClose'].isnull().all() or options_t1['DlyClose'].isnull().all(): # If there is no close at t = 0 or t = 1 we can't calculate anything
                    self.save_nan_values(t, q_string)
                    continue


                # Get the closing prices at t=0 and t =1
                close_t0 = df_[df_['t'] == 0]['DlyClose'].iloc[0]
                close_t1 = df_[df_['t'] == 1]['DlyClose'].iloc[0]


                vix_t0 = self.vix.loc[date_t0, 'vix'] # Get the Vix at t = 0

                log_marketcap_t0 = np.log(options_t0['DlyCap'].values[0] * 1000) # Get the log market cap at t = 0

                if log_marketcap_t0 <= np.log(1_000_000_000): # We do not care about companies smaller thna 1 bln
                    self.save_nan_values(t, q_string)
                    continue

                # Get the closing values of the stock in the past 42 trading days
                date_t0_2months_ago = date_t0_ts - pd.DateOffset(months = 2)
                stock_ticker = stock_ticker_df[(stock_ticker_df['DlyCalDt'] >= date_t0_2months_ago) & (stock_ticker_df['DlyCalDt'] <=date_t0)].copy()
                # We will use it to calculate the beta to the overall market
                stock_ticker['Return_Stock'] = (stock_ticker['DlyClose'] - stock_ticker['DlyClose'].shift(1) ) / stock_ticker['DlyClose'].shift(1) # Calculate the Return of the Stock
                stock_ticker = stock_ticker.merge(self.sp500[['DlyCalDt', 'Return']], left_on = 'DlyCalDt', right_on = 'DlyCalDt', how = 'left') # Merge with the SP 500 Returns
                stock_ticker = stock_ticker.dropna(subset = ['Return']) # Drop NAs
                stock_ticker = stock_ticker.dropna(subset = ['Return_Stock']) # Drop NAs
                beta = stock_ticker['Return_Stock'].cov(stock_ticker['Return']) / stock_ticker['Return'].var() # Beta = Cov(X, Y) / Var(X)


                realized_move = np.abs(close_t1 - close_t0)
                realized_move_pct = (close_t1 - close_t0)/close_t0
                realized_move_pct_abs = np.abs( (close_t1 - close_t0)/close_t0 )
                

                # Get the front and back month DTE at t = 0, will be used to isolate the option chains for the implied moves etc..
                front_month = min(options_t0['DTE'].unique().tolist())
                options_t0_front_month = options_t0[options_t0['DTE'] == front_month] # This is the Front Month Option chain at t = 0
                front_month_date = options_t0_front_month['exdate'].values[0] # Get the date of the Front Month Expiration to be able to track it over time

                # Consider only options where we have strikes of the same price
                strike_counts = options_t0_front_month['strike_price'].value_counts()
                valid_strikes = strike_counts[strike_counts == 2].index

                # Get all the strike pairs that have both a Put and a Call for them
                mask = options_t0_front_month['strike_price'].isin(valid_strikes)
                options_t0_front_month_complete_pairs = options_t0_front_month[mask]

                if self.verbose:
                    print(options_t0)
                    print(options_t1)

                strike_prices = pd.Series(options_t0_front_month_complete_pairs['strike_price'].unique())

                if self.verbose:
                    print(f'Cose t0 {close_t0}, strike prices {strike_prices}')

                atm_strike = strike_prices.iloc[(strike_prices - close_t0).abs().idxmin()]

                # Get the Call and Put at t = 0
                call_t0 = options_t0_front_month[(options_t0_front_month['strike_price'] == atm_strike) & (options_t0_front_month['cp_flag'] == 'C')]
                put_t0 = options_t0_front_month[(options_t0_front_month['strike_price'] == atm_strike) & (options_t0_front_month['cp_flag'] == 'P')]

                front_month_iv = (call_t0['impl_volatility'].values[0] + put_t0['impl_volatility'].values[0]) / 2 # This will also be a feature!!!!!!!!
                atm_call_t0_open_interest_ratio = call_t0['open_interest'].values[0] / options_t0_front_month['open_interest'].sum() # This will be a feature!!!! The proportion of all open interest at the ATM call
                atm_put_t0_open_interest_ratio = put_t0['open_interest'].values[0] / options_t0_front_month['open_interest'].sum() # This will be a feature !!!! The proportion of all open interest compared to the open interest of the Straddle Put
                call_put_ratio = options_t0_front_month[options_t0_front_month['cp_flag'] == 'C']['open_interest'].sum() / options_t0_front_month[options_t0_front_month['cp_flag'] == 'P']['open_interest'].sum()
                # The final features will be the growth in volume relative to previous days in %
                df__ = df_[df_['exdate'] == front_month_date]
                hist_mask_c = (df__['t'] >= -5) & (df__['t'] <= -1) & (df__['cp_flag'] == 'C')# Get the calls from t = -5 to t = -1 
                hist_mask_p = (df__['t'] >= -5) & (df__['t'] <= -1) & (df__['cp_flag'] == 'P') # Get the puts from t = -5 to t = -1

                hist_mean_calls = df__.loc[hist_mask_c, 'volume'].mean() # Calculate the mean call volume
                hist_mean_puts = df__.loc[hist_mask_p, 'volume'].mean() # Calculate the mean put volume

                curr_mean_calls = options_t0.loc[options_t0['cp_flag'] == 'C', 'volume'].mean() 
                curr_mean_puts = options_t0.loc[options_t0['cp_flag'] == 'P', 'volume'].mean() 

                abnormal_volume_calls = (curr_mean_calls - hist_mean_calls) / hist_mean_calls - 1 if hist_mean_calls > 0 else 0# These will be features
                abnormal_volume_puts = (curr_mean_puts - hist_mean_puts) / hist_mean_puts  - 1 if hist_mean_puts > 0 else 0# These will be features


                if self.verbose:
                    print(f' ATM Call Open Interest Ratio {round(atm_call_t0_open_interest_ratio, 3)}, ATM Put Open Interest Ratio {round(atm_put_t0_open_interest_ratio,3)}')
                
                # If there is a backmonth we will calculate the Expected Move using the Bloomberg Formula
                if len(options_t0['DTE'].unique().tolist()) >= 2:
                    back_month = sorted(options_t0['DTE'].unique().tolist())[1]
                    options_t0_back_month = options_t0[options_t0['DTE'] == back_month]
                    
                    available_back_strikes = options_t0_back_month['strike_price'].unique() # Get all the strikes in the back month


                    if len(available_back_strikes) > 0: # If there are strikes in the back month which there should be

                        closest_back_strike = available_back_strikes[np.abs(available_back_strikes - atm_strike).argmin()]

                        backmonth_t0 = options_t0_back_month[options_t0_back_month['strike_price'] == closest_back_strike]

                        call_iv_back_month = backmonth_t0[backmonth_t0['cp_flag'] == 'C']['impl_volatility'].values[0]
                        put_iv_back_month = backmonth_t0[backmonth_t0['cp_flag'] == 'P']['impl_volatility'].values[0]

                        back_month_iv = (call_iv_back_month + put_iv_back_month) / 2
                        implied_move = self.event_expected_move(front_month_iv, back_month_iv, front_month, back_month, 1)
                        iv_slope = (front_month_iv - back_month_iv) / (back_month - front_month)
                    
                    else:
                        implied_move = self.event_expected_move_front_vol(front_month_iv)
                        iv_slope = np.nan

                        
                else: # if not backmonth present there is no IV slope and we calculate the implied move using the event variance to be that from the front month option chain
                    implied_move = self.event_expected_move_front_vol(front_month_iv)
                    iv_slope = np.nan

                # Calculate an ATM Straddle Bid and Ask at t = 0
                straddle_bid_t0 = call_t0['best_bid'].values[0] + put_t0['best_bid'].values[0]
                straddle_offer_t0 = call_t0['best_offer'].values[0] + put_t0['best_offer'].values[0]

                bid_ask_straddle_premium = straddle_bid_t0 # Getting filled in a Straddle at the best bid
                realistic_straddle_premium = straddle_bid_t0 + 0.25 * (straddle_offer_t0 - straddle_bid_t0) if straddle_offer_t0 >= straddle_bid_t0 else straddle_bid_t0 # Realistic fill at 25% of the bid-ask spread

                call_identifier = call_t0['optionid'].values[0]
                put_identifier = put_t0['optionid'].values[0]

                # Sometimes probably we will have to delta hedge upon entry especially if the stock < 20$
                if front_month >= 2: # Only Delta Hedge is more than 2 DTE
                    delta_difference = round(100 * (call_t0['delta'].values[0] + put_t0['delta'].values[0]),0) if isinstance(call_t0['delta'].values[0], float) and isinstance(put_t0['delta'].values[0], float) else 0
                    delta_hedge_pnl = - delta_difference * realized_move
                    if np.abs(delta_difference) >= 20: # We won't hedge anything where the delta difference is more than 20 this becomes way too crazy
                        delta_hedge_pnl = 0

                else:
                    delta_difference = 0
                    delta_hedge_pnl = 0

                

                # Get the options from t = 0 after the event at t = 1
                call_t1 = options_t1[options_t1['optionid'] == call_identifier]
                put_t1 = options_t1[options_t1['optionid'] == put_identifier]

                # If the call or put at t = 1 are empty than we just save all features as NaN
                if call_t1.empty or put_t1.empty:
                    self.save_nan_values(t, q_string)
                    continue

                # Get the bid and ask of the straddle which was entered at t = 0 after the event at t = 1
                straddle_offer_t1 = call_t1['best_offer'].values[0] + put_t1['best_offer'].values[0]
                straddle_bid_t1 = call_t1['best_bid'].values[0] + put_t1['best_bid'].values[0]

                bid_ask_straddle_debit = straddle_offer_t1
                realistic_straddle_debit = straddle_offer_t1 - 0.25 * (straddle_offer_t1 - straddle_bid_t1) if straddle_offer_t1 >= straddle_bid_t1 else straddle_offer_t1

                # If the spread is bigger than 2 times the bid size we simply will not trade 
                if call_t0['best_bid'].values[0] < 2 * (call_t0['best_offer'].values[0] - call_t0['best_bid'].values[0]) or put_t0['best_bid'].values[0] < 2 * (put_t0['best_offer'].values[0] - put_t0['best_bid'].values[0]):
                    self.save_nan_values(t, q_string)
                    continue



                bid_ask_straddle_pnl = (bid_ask_straddle_premium - bid_ask_straddle_debit + delta_hedge_pnl / 100) / bid_ask_straddle_premium if delta_hedge_pnl != 0 else (bid_ask_straddle_premium - bid_ask_straddle_debit)/bid_ask_straddle_premium
                realistic_straddle_pnl = (realistic_straddle_premium - realistic_straddle_debit + delta_hedge_pnl / 100) / realistic_straddle_premium if delta_hedge_pnl !=0 else (realistic_straddle_premium - realistic_straddle_debit)/realistic_straddle_premium

                if bid_ask_straddle_pnl > 1: # In very rare cases are we able to Gt above Straddle PNL of 100% that means that the options expired precisely worthless on a friday and we moved up with a positive delta hedge co
                    print(f'Credit {bid_ask_straddle_premium}, Debit {bid_ask_straddle_debit}')
                    print(call_t0)
                    print(call_t1)
                    print(put_t0)
                    print(put_t1)
                    print(delta_hedge_pnl / 100)


                if self.verbose:
                    print(call_t0)
                    print(put_t0)
                    print(call_t1)
                    print(put_t1)

                if front_month == 1: # If we sold the option 1 day before expiration at t = 0 than the option would not show up in the t1 so
                    bid_ask_straddle_pnl = (bid_ask_straddle_premium - realized_move + delta_hedge_pnl / 100) / bid_ask_straddle_premium
                    realistic_straddle_pnl = (realistic_straddle_premium - realized_move + delta_hedge_pnl / 100) / realistic_straddle_premium

                df_t15 = df__[df__['t'] == -15] # Options at t = -15 for the first expiration after the event
                df_t5 = df__[df__['t'] == -5] # Options at t = -5 for the first expiration after the event

                if len(df_t15['date'].values) == 0 or len(df_t5['date'].values) == 0 or len(df_[df_['t'] == -15]['DlyClose'].values) == 0 or len(df_[df_['t'] == -5]['DlyClose'].values) == 0: # No implied skew and kurtosis calculation if no data present for t = -5 or t = -15
                    skew_t15, kurt_t15 = np.nan, np.nan
                    skew_t5, kurt_t5 = np.nan, np.nan

                else:
                    close_t15 = df_[df_['t'] == -15]['DlyClose'].iloc[0] # Close ast t = -15
                    close_t5 = df_[df_['t'] == -5]['DlyClose'].iloc[0] # Close ast t = -5

                    date_t15 =  df_t15['date'].values[0]# the date of t = -15
                    date_t5 = df_t5['date'].values[0] # the date of t = -5

                    try:
                        rf_t15 = self.risk_free.loc[date_t15, 'RF']# RF at t = -15
                    except:
                        try:
                            rf_t15 = self.risk_free.loc[date_t15 - pd.Timedelta(days = 1), 'RF']# RF at t = -15
                        except:
                            rf_t15 = self.risk_free.loc[date_t15 + pd.Timedelta(days = 1), 'RF']# RF at t = -15

                    try:
                        rf_t5 = self.risk_free.loc[date_t5, 'RF']# RF at t = -5
                    except:
                        try:
                            rf_t5 = self.risk_free.loc[date_t5 - pd.Timedelta(days = 1), 'RF']# RF at t = -5
                        except:
                            rf_t5 = self.risk_free.loc[date_t5 + pd.Timedelta(days = 1), 'RF']# RF at t = -5


                    t_15_years = df_t15['DTE'].values[0] / 365 # t in years at t = -15
                    t_5_years = df_t5['DTE'].values[0] / 365 # t in years at t = -15

                    skew_t15, kurt_t15 = self.calculate_model_free_stats(df_t15, close_t15, rf_t15, t_15_years)
                    skew_t5, kurt_t5 = self.calculate_model_free_stats(df_t5, close_t5, rf_t5, t_5_years)

                # If some skew is NaN also delta skew is Nan
                if np.isnan(skew_t15) or np.isnan(skew_t5):
                    delta_skew = np.nan
                else:
                    delta_skew = skew_t15 - skew_t5
                
                # If some kurt is NaN also delta kurt is Nan
                if np.isnan(kurt_t15) or np.isnan(kurt_t5):
                    delta_kurt = np.nan        
                else:
                    delta_kurt = kurt_t15 - kurt_t5


                print(f'Implied Skew t = -15 {round(skew_t15, 3)}, Kurt {round(kurt_t15, 3)}. Implied Skew t = -5 {round(skew_t5,3)}, Kurt {round(kurt_t5,3)}')
                print(f'Front Month DTE {front_month}, Back Month DTE {back_month}, straddle pnl {round(bid_ask_straddle_pnl * 100, 2)}%, {delta_difference} Delta were hedged for a delta hedge PNL {round( 100 * delta_hedge_pnl / 100 / bid_ask_straddle_premium, 2)}%, pure straddle pnl was {round( 100 * (bid_ask_straddle_premium - bid_ask_straddle_debit)/bid_ask_straddle_premium ,2)}%')


                self.abnormal_volume_call.loc[t, q_string] = abnormal_volume_calls 
                self.abnormal_volume_put.loc[t, q_string] = abnormal_volume_puts
                self.after_market.loc[t, q_string] = call_t0['After Market'].values[0]
                self.atm_call_open_interest_ratio.loc[t, q_string] = atm_call_t0_open_interest_ratio
                self.atm_put_open_interest_ratio.loc[t, q_string] = atm_put_t0_open_interest_ratio
                self.beta.loc[t, q_string] = beta
                self.call_put_ratio.loc[t, q_string] = call_put_ratio
                self.day_of_week.loc[t, q_string] = call_t0['day_week'].values[0]
                self.implied_kurt_change.loc[t, q_string] =  delta_kurt
                self.implied_skew_change.loc[t, q_string] = delta_skew 
                self.implied_move.loc[t, q_string] = implied_move
                self.implied_vol.loc[t, q_string] = front_month_iv
                self.iv_slope.loc[t, q_string] = iv_slope
                self.log_market_cap.loc[t, q_string] = log_marketcap_t0 
                self.realized_move_pct.loc[t, q_string] = realized_move_pct
                self.realized_move_pct_abs.loc[t, q_string] = realized_move_pct_abs
                self.vix_.loc[t, q_string] = vix_t0
                self.date.loc[t, q_string] = date_t0_ts.strftime('%Y-%m-%d')
                self.pnl_bid_ask_df.loc[t, q_string] = bid_ask_straddle_pnl
                self.pnl_realistic_df.loc[t, q_string] = realistic_straddle_pnl



        self.abnormal_volume_call.to_csv(r'D:\Option Data\feature_matrixes\abnormal_volume_call.csv', index = True)
        self.abnormal_volume_put.to_csv(r'D:\Option Data\feature_matrixes\abnormal_volume_put.csv', index = True)
        self.after_market.to_csv(r'D:\Option Data\feature_matrixes\after_market.csv', index = True)
        self.atm_call_open_interest_ratio.to_csv(r'D:\Option Data\feature_matrixes\atm_call_open_interest_ratio.csv',index = True)
        self.atm_put_open_interest_ratio.to_csv(r'D:\Option Data\feature_matrixes\atm_put_open_interest_ratio.csv',index = True)
        self.beta.to_csv(r'D:\Option Data\feature_matrixes\beta.csv',index = True)
        self.call_put_ratio.to_csv(r'D:\Option Data\feature_matrixes\call_put_ratio.csv',index = True)
        self.day_of_week.to_csv(r'D:\Option Data\feature_matrixes\day_of_week.csv',index = True)
        self.implied_kurt_change.to_csv(r'D:\Option Data\feature_matrixes\implied_kurt_change.csv',index = True)
        self.implied_skew_change.to_csv(r'D:\Option Data\feature_matrixes\implied_skew_change.csv',index = True)
        self.implied_move.to_csv(r'D:\Option Data\feature_matrixes\implied_move.csv',index = True)
        self.implied_vol.to_csv(r'D:\Option Data\feature_matrixes\implied_vol.csv',index = True)
        self.iv_slope.to_csv(r'D:\Option Data\feature_matrixes\iv_slope.csv',index = True)
        self.log_market_cap.to_csv(r'D:\Option Data\feature_matrixes\log_market_cap.csv',index = True)
        self.realized_move_pct.to_csv(r'D:\Option Data\feature_matrixes\realized_move_pct.csv',index = True)
        self.realized_move_pct_abs.to_csv(r'D:\Option Data\feature_matrixes\realized_move_pct_abs.csv',index = True)
        self.vix_.to_csv(r'D:\Option Data\feature_matrixes\vix.csv',index = True)
        self.date.to_csv(r'D:\Option Data\feature_matrixes\date.csv',index = True)
        self.pnl_bid_ask_df.to_csv(r'D:\Option Data\feature_matrixes\pnl_bid_ask_backtest.csv',index = True)
        self.pnl_realistic_df.to_csv(r'D:\Option Data\feature_matrixes\pnl_realistic_backtest.csv',index = True)
                

    def event_expected_move(self, iv1, iv2, t1, t2, te):

        try:
            event_before_t1 = 1 if te <= t1 else 0
            event_before_t2 = 1 if te <= t2 else 0

            A = np.array([[event_before_t1, t1 - event_before_t1], [event_before_t2, t2 - event_before_t2]])
            b = np.array([t1 * (iv1 ** 2), t2 * (iv2 ** 2)])
            result = np.linalg.solve(A, b)

            event_variance = result[0]
            event_vol = event_variance ** 0.5
            expected_move = np.sqrt(2 / np.pi) * event_vol / (252 ** 0.5)
            return expected_move
        except:
            return np.nan
    
    def event_expected_move_front_vol(self, iv1):
        expected_move = np.sqrt(2/np.pi) * iv1 / (252 ** 0.5)


    def calculate_model_free_stats(self, options_df, S, r, T):
        """
        Implements Bakshi, Kapadia, Madan (2003) for discrete strikes.
        options_df: DataFrame with columns ['strike_price', 'cp_flag', 'best_bid', 'best_offer']
        S: Current Stock Price (close_t0)
        r: Risk-free rate (e.g., 0.05)
        T: Time to maturity in years (DTE / 365)
        """
        # 1. Prepare data: use mid-price and sort by strike
        df = options_df.copy()
        df['price'] = (df['best_bid'] + df['best_offer']) / 2
        df = df.sort_values('strike_price')
        
        calls = df[df['cp_flag'] == 'C']
        puts = df[df['cp_flag'] == 'P']

        if len(calls[calls['strike_price'] > S]) < 3 or len(puts[puts['strike_price'] < S]) < 3:
            return np.nan, np.nan
        
        # 2. Numerical Integration (Trapezoidal rule)
        def integrate_contract(contract_type):
            val = 0
            # Sum over OTM Calls (K > S)
            otm_calls = calls[calls['strike_price'] > S]
            for i in range(1, len(otm_calls)):
                K = otm_calls.iloc[i]['strike_price']
                C = otm_calls.iloc[i]['price']
                dK = K - otm_calls.iloc[i-1]['strike_price']
                
                if contract_type == 'V':
                    f_K = (2 * (1 - np.log(K/S))) / (K**2)
                elif contract_type == 'W':
                    f_K = (6 * np.log(K/S) - 3 * (np.log(K/S)**2)) / (K**2)
                elif contract_type == 'X':
                    f_K = (12 * (np.log(K/S)**2) - 4 * (np.log(K/S)**3)) / (K**2)
                
                val += f_K * C * dK

            # Sum over OTM Puts (K < S)
            otm_puts = puts[puts['strike_price'] < S]
            for i in range(1, len(otm_puts)):
                K = otm_puts.iloc[i]['strike_price']
                P = otm_puts.iloc[i]['price']
                dK = otm_puts.iloc[i]['strike_price'] - otm_puts.iloc[i-1]['strike_price']
                
                if contract_type == 'V':
                    f_K = (2 * (1 + np.log(S/K))) / (K**2)
                elif contract_type == 'W':
                    f_K = (6 * np.log(S/K) + 3 * (np.log(S/K)**2)) / (K**2)
                elif contract_type == 'X':
                    f_K = (12 * (np.log(S/K)**2) + 4 * (np.log(S/K)**3)) / (K**2)
                    
                val += f_K * P * dK
            return val

        # Calculate V, W, X from Equation (A.3, A.4, A.5)
        V = integrate_contract('V')
        W = integrate_contract('W')
        X = integrate_contract('X')
        
        # Calculate mu from Equation (A.6)
        exp_rt = np.exp(r * T)
        mu = exp_rt - 1 - (exp_rt/2 * V) - (exp_rt/6 * W) - (exp_rt/24 * X)
        

        if (exp_rt * V - mu**2) <= 0:
            return np.nan, np.nan

        # Calculate SKEW (A.1) and KURT (A.2)
        # Using the second part of the equalities in the provided image
        skew_num = (exp_rt * W) - (3 * mu * exp_rt * V) + (2 * mu**3)
        skew_den = (exp_rt * V - mu**2)**(3/2)
        skew = skew_num / skew_den if skew_den != 0 else np.nan
        
        kurt_num = (exp_rt * X) - (4 * mu * exp_rt * W) + (6 * exp_rt * mu**2 * V) - (3 * mu**4)
        kurt_den = (exp_rt * V - mu**2)**2
        kurt = (kurt_num / kurt_den) if kurt_den != 0 else np.nan
        
        return skew, kurt


    def save_nan_values(self, ticker, quarter):
        self.abnormal_volume_call.loc[ticker, quarter] = np.nan 
        self.abnormal_volume_put.loc[ticker, quarter] = np.nan 
        self.after_market.loc[ticker, quarter] = np.nan 
        self.atm_call_open_interest_ratio.loc[ticker, quarter] = np.nan 
        self.atm_put_open_interest_ratio.loc[ticker, quarter] = np.nan 
        self.beta.loc[ticker, quarter] = np.nan 
        self.call_put_ratio.loc[ticker, quarter] = np.nan 
        self.day_of_week.loc[ticker, quarter] = np.nan 
        self.implied_kurt_change.loc[ticker, quarter] =  np.nan
        self.implied_skew_change.loc[ticker, quarter] = np.nan 
        self.implied_move.loc[ticker, quarter] = np.nan 
        self.implied_vol.loc[ticker, quarter] = np.nan 
        self.iv_slope.loc[ticker, quarter] = np.nan 
        self.log_market_cap.loc[ticker, quarter] = np.nan 
        self.realized_move_pct.loc[ticker, quarter] = np.nan 
        self.realized_move_pct_abs.loc[ticker, quarter] = np.nan 
        self.vix_.loc[ticker, quarter] = np.nan 
        self.date.loc[ticker, quarter] = np.nan
        self.pnl_bid_ask_df.loc[ticker, quarter] = np.nan
        self.pnl_realistic_df.loc[ticker, quarter] = np.nan


class FlattenFeatures():

    def __init__(self):
        self.abnormal_volume_call = pd.read_csv(r'D:\Option Data\feature_matrixes\abnormal_volume_call.csv', index_col =0)
        self.abnormal_volume_put = pd.read_csv(r'D:\Option Data\feature_matrixes\abnormal_volume_put.csv', index_col =0)
        self.after_market = pd.read_csv(r'D:\Option Data\feature_matrixes\after_market.csv', index_col =0)
        self.atm_call_open_interest_ratio = pd.read_csv(r'D:\Option Data\feature_matrixes\atm_call_open_interest_ratio.csv', index_col =0)
        self.atm_put_open_interest_ratio = pd.read_csv(r'D:\Option Data\feature_matrixes\atm_put_open_interest_ratio.csv', index_col = 0)
        self.beta = pd.read_csv(r'D:\Option Data\feature_matrixes\beta.csv', index_col = 0)
        self.call_put_ratio = pd.read_csv(r'D:\Option Data\feature_matrixes\call_put_ratio.csv', index_col = 0)
        self.day_of_week = pd.read_csv(r'D:\Option Data\feature_matrixes\day_of_week.csv', index_col = 0)
        self.implied_kurt_change = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_kurt_change.csv', index_col = 0)
        self.implied_skew_change = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_skew_change.csv', index_col = 0)
        self.implied_move = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_move.csv', index_col = 0)
        self.implied_vol = pd.read_csv(r'D:\Option Data\feature_matrixes\implied_vol.csv', index_col = 0)
        self.iv_slope = pd.read_csv(r'D:\Option Data\feature_matrixes\iv_slope.csv', index_col = 0)
        self.log_market_cap = pd.read_csv(r'D:\Option Data\feature_matrixes\log_market_cap.csv', index_col = 0)
        self.realized_move_pct = pd.read_csv(r'D:\Option Data\feature_matrixes\realized_move_pct.csv', index_col = 0)
        self.realized_move_pct_abs = pd.read_csv(r'D:\Option Data\feature_matrixes\realized_move_pct_abs.csv', index_col = 0)
        self.vix_ = pd.read_csv(r'D:\Option Data\feature_matrixes\vix.csv', index_col = 0)
        self.date = pd.read_csv(r'D:\Option Data\feature_matrixes\date.csv', index_col = 0, dtype=object)
        self.pnl_bid_ask_df = pd.read_csv(r'D:\Option Data\feature_matrixes\pnl_bid_ask_backtest.csv', index_col = 0)
        self.pnl_realistic_df = pd.read_csv(r'D:\Option Data\feature_matrixes\pnl_realistic_backtest.csv', index_col = 0)
        

        self.df_X = pd.DataFrame(columns=['Ticker', 'Q-String','Abnormal Volume Call', 'Abnormal Volume Put', 'After Market', 'ATM Call Open Interest Ratio', 'ATM Put Open Interest Ratio', 'Beta', 'Call Put Ratio', 'Day Of Week', 'Kurt Delta', 'Skew Delta', 'Implied Move', 'Implied Vol', 'IV Slope', 'Log Market Cap', 'Vix', 'Date'])
        self.df_X_auxiliary = pd.DataFrame(columns = ['Ticker', 'Q-String', 'Realized Move Pct', 'Realized Move Pct Abs', 'PNL Bid-Ask', 'PNL Realistic'])


        for index, row in self.date.iterrows():

            for col in self.date.columns:
                print(row[col])
                print(f'Index is {index}, col is {col}')


if __name__ == '__main__':
    #PartionOptionData()
    #PreprocessOptionData()
    #CalculateFeatureMatrixes(verbose= False)
    FlattenFeatures()
